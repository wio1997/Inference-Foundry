#!/usr/bin/env python3
"""Extract fixed F16 CSV fields for 48 previously anchored pairs, no PMU join."""
import argparse, csv, hashlib, json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--task-root', required=True)
ap.add_argument('--trace-root', required=True)
ap.add_argument('--anchors-json', required=True)
a = ap.parse_args()
task, trace = Path(a.task_root), Path(a.trace_root)
anchors = json.loads(a.anchors_json)
assert anchors['sha256'] == '26d0ab676539995816c26f697d0702941e7284983f020ff9f221f86c3d81840f'
wp = task / 'captured-layer-attribution/module-call-metrics.json'
manifest = task / 'evidence/TRACE-EXPORT-SHA256.json'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
assert sha(wp) == 'd98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'
assert sha(manifest) == '45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'
expected = {r['rank']: r['kernel_csv_sha256'] for r in json.loads(manifest.read_text())['exports']}
windows = {(w['rank'],w['run_in_model'],w['layer']): w for w in json.loads(wp.read_text())}
assert len(anchors['rows']) == 48 and len(expected) == 16
by_rank = defaultdict(list)
for r in anchors['rows']:
    assert r['layer'] == 24 and r['run'] in (1,2,3) and r['unique_ordered_candidates']
    w = windows[(r['rank'],r['run'],r['layer'])]
    assert str(w['model_id']) == r['model'] == '49' and str(w['main_stream_id']) == r['main_stream']
    for label in ('first','second'):
        assert len(r[label]) == 1
        e = r[label][0]
        assert 0 <= e['start_rel_ns'] < e['end_rel_ns'] <= w['window_end_ns'] - w['window_start_ns']
        by_rank[r['rank']].append({'rank': r['rank'], 'run': r['run'], 'label': label, 'model': r['model'],
            'stream': r['main_stream'], 'start': w['window_start_ns']+e['start_rel_ns'],
            'end': w['window_start_ns']+e['end_rel_ns'], 'original': e})
assert set(by_rank) == set(expected) and all(len(v) == 6 for v in by_rank.values())
fields = ['Name','Type','Task ID','Accelerator Core','Block Num','Mix Block Num','OP State',
          'Input Formats','Output Formats','Input Shapes','Output Shapes','Input Data Types','Output Data Types',
          'Wait Time(us)','aic_total_cycles','aic_mac_ratio','aic_scalar_ratio','aic_mte1_ratio','aic_mte2_ratio',
          'aic_fixpipe_ratio','aic_icache_miss_rate','aiv_total_cycles','aiv_vec_ratio','aiv_scalar_ratio',
          'aiv_mte2_ratio','aiv_mte3_ratio','aiv_icache_miss_rate','cube_utilization(%)']
rows=[]
for rank, targets in sorted(by_rank.items()):
    csv_path = trace / rank / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
    assert csv_path.stat().st_size <= 50*1024*1024 and sha(csv_path) == expected[rank]
    hits = {(z['run'],z['label']): [] for z in targets}
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        assert {'Type','Model ID','Stream ID','Start Time(us)','Duration(us)'}.issubset(reader.fieldnames or ())
        missing = [k for k in fields if k not in (reader.fieldnames or ())]
        for z in reader:
            if z['Type'] not in ('TransposeBatchMatMul','MatMulV2'):
                continue
            start = int(Decimal(z['Start Time(us)'].strip())*1000)
            end = start + int(Decimal(z['Duration(us)'].strip())*1000)
            for target in targets:
                e = target['original']
                if (start,end,z['Model ID'].strip(),z['Stream ID'].strip(),z['Type'],z['Task ID'].strip()) != (target['start'],target['end'],target['model'],target['stream'],e['type'],e['task_id'].strip()):
                    continue
                assert z['Input Shapes'].strip().strip('"') == e['input_shapes'] and z['Output Shapes'].strip().strip('"') == e['output_shapes']
                assert z['Input Data Types'].strip() == e['input_dtypes'] and z['Output Data Types'].strip() == e['output_dtype']
                hits[(target['run'],target['label'])].append({'rank':rank,'run':target['run'],'label':target['label'],
                    'start_ns':start,'end_ns':end,'model':target['model'],'stream':target['stream'],
                    'fields':{k:z.get(k) for k in fields},'missing_columns':missing})
    for key, matches in sorted(hits.items()):
        assert len(matches) == 1, 'exact CSV target not unique'
        rows.append(matches[0])
assert len(rows) == 96 and len({(r['rank'],r['run'],r['label']) for r in rows}) == 96
print(json.dumps({'scope':'96 exact F16 CSV rows, 16 rank x 3 fixed layer24 windows x 2 labels; no raw PMU join',
 'anchors_sha256':anchors['sha256'],'window_sha256':sha(wp),'manifest_sha256':sha(manifest),'csv_hashes':expected,
 'fields_requested':fields,'samples':rows,'limits':['CSV exact event identity is not proof of hardware counter exporter replay mapping or binary provenance.',
 'Ratios are reported decimals with unknown denominator, aggregation, accuracy and rounding; no hardware utilization, FLOP or bytes inferred.',
 'Missing or zero values are not evidence of zero device activity; no raw PMU, cross-rank timestamps, request/tensor identity or NPU operation.']},ensure_ascii=False,indent=2))
