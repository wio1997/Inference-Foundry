#!/usr/bin/env python3
"""Bounded rank-local overlap in 16 saved CSVs. Does not invoke NPU/runtime."""
import argparse, csv, hashlib, json, statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
task=Path(a.task_root); traces=Path(a.trace_root)
wp=task/'captured-layer-attribution/module-call-metrics.json';manifest=task/'evidence/TRACE-EXPORT-SHA256.json'
expected={r['rank']:r['kernel_csv_sha256'] for r in json.loads(manifest.read_text())['exports']}
byrank=defaultdict(list)
for w in json.loads(wp.read_text()):
    if w['layer']==0 and w['run_in_model'] in (1,2,3):byrank[w['rank']].append(w)
assert len(byrank)==16 and set(byrank)==set(expected)
for rank,ws in byrank.items():
    assert len(ws)==3 and {w['run_in_model'] for w in ws}=={1,2,3}
    assert len({str(w['model_id']) for w in ws})==1 and len({str(w['main_stream_id']) for w in ws})==1
    ws.sort(key=lambda w:w['window_start_ns'])
    assert all(x['window_end_ns']<=y['window_start_ns'] for x,y in zip(ws,ws[1:]))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def union(spans):
    merged=[]
    for lo,hi in sorted(spans):
        if lo>=hi:continue
        if not merged or lo>merged[-1][1]:merged.append([lo,hi])
        else:merged[-1][1]=max(merged[-1][1],hi)
    return merged
def length(spans):return sum(b-a for a,b in union(spans))
def summarize(rows):
    types=defaultdict(list);streams=defaultdict(list);models=defaultdict(list)
    for r in rows:
        span=(r['start_ns'],r['end_ns']);types[r['type']].append(span);streams[r['stream']].append(span);models[r['model']].append(span)
    return {'rows':len(rows),'union_ns':length([(r['start_ns'],r['end_ns']) for r in rows]),
            'types':{k:{'rows':len(v),'union_ns':length(v)} for k,v in sorted(types.items())},
            'streams':{k:{'rows':len(v),'union_ns':length(v)} for k,v in sorted(streams.items())},
            'models':{k:{'rows':len(v),'union_ns':length(v)} for k,v in sorted(models.items())}}

rows=[];actual={}
for rank,ws in sorted(byrank.items()):
    p=traces/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'; actual[rank]=sha(p)
    assert actual[rank]==expected[rank],f'CSV hash mismatch {rank}'
    events={w['run_in_model']:[] for w in ws}
    with p.open() as f:
        for z in csv.DictReader(f):
            start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
            if end<=start:continue
            for w in ws:
                if start<w['window_end_ns'] and end>w['window_start_ns']:
                    events[w['run_in_model']].append({'start_ns':start,'end_ns':end,'type':z['Type'],
                        'stream':z['Stream ID'].strip(),'model':z['Model ID'].strip()})
    for w in ws:
        ev=events[w['run_in_model']]; main=str(w['main_stream_id']);model=str(w['model_id'])
        own=[e for e in ev if e['type']=='DispatchFFNCombine' and e['stream']==main and e['model']==model
             and w['window_start_ns']<=e['start_ns'] and e['end_ns']<=w['window_end_ns']]
        assert len(own)==1,f'missing/ambiguous MC2 {rank} run {w["run_in_model"]} {len(own)}'
        m=own[0];dur=m['end_ns']-m['start_ns'];other=[];unknown=[]
        for e in ev:
            if e['start_ns']>=m['end_ns'] or e['end_ns']<=m['start_ns'] or e['stream']==main:continue
            clipped={**e,'start_ns':max(e['start_ns'],m['start_ns']),'end_ns':min(e['end_ns'],m['end_ns'])}
            (other if e['stream'].isdigit() else unknown).append(clipped)
        assert len(other)+len(unknown)<=500, 'output bound exceeded; inspect explicitly'
        allgather=[e for e in other if e['type']=='allgatherAicpuKernel']
        hcom=[e for e in other if e['type'].lower().startswith('hcom_')]
        summary=summarize(other);unknown_summary=summarize(unknown)
        assert 0<=summary['union_ns']<=dur and 0<=unknown_summary['union_ns']<=dur
        rows.append({'rank':rank,'run':w['run_in_model'],'model':model,'main_stream':main,
            'mc2_start_ns':m['start_ns'],'mc2_end_ns':m['end_ns'],'mc2_duration_ns':dur,
            'other_numeric':summary,'allgather_aicpu':summarize(allgather),'hcom_named':summarize(hcom),
            'unknown_stream':unknown_summary,'numeric_coverage_gap_ns':dur-summary['union_ns'],
            'clipped_other_numeric_rows':other,'clipped_unknown_stream_rows':unknown})
assert len(rows)==48
summary=[]
for run in (1,2,3):
    group=[r for r in rows if r['run']==run];assert len(group)==16
    frac=lambda r,k:r[k]['union_ns']/r['mc2_duration_ns']
    summary.append({'run':run,'ranks':16,
      'mc2_duration_us_median':statistics.median(r['mc2_duration_ns']/1000 for r in group),
      'other_numeric_nonzero_ranks':sum(r['other_numeric']['union_ns']>0 for r in group),
      'other_numeric_fraction_min':min(frac(r,'other_numeric') for r in group),
      'other_numeric_fraction_median':statistics.median(frac(r,'other_numeric') for r in group),
      'other_numeric_fraction_max':max(frac(r,'other_numeric') for r in group),
      'allgather_aicpu_nonzero_ranks':sum(r['allgather_aicpu']['union_ns']>0 for r in group),
      'allgather_aicpu_fraction_median':statistics.median(frac(r,'allgather_aicpu') for r in group),
      'unknown_stream_nonzero_ranks':sum(r['unknown_stream']['union_ns']>0 for r in group)})
out={'scope':'16 ranks x 3 same-model, same-main-stream attributed layer0 MC2 samples; rank-local overlap of exported kernel CSV only',
     'window_sha256':sha(wp),'manifest_sha256':sha(manifest),'kernel_sha256_by_rank':actual,'samples':rows,'run_summary':summary,
     'limitations':['Numeric other-stream rows may represent another request/model; temporal overlap does not imply dependency or attribution.',
       'allgatherAicpuKernel and hcom_ are exported names, not proof of link transfer, waiting or communication hidden by computation.',
       'Unknown stream rows are kept separate: they may be synthetic or duplicate. Different type/stream unions may overlap; do not add.',
       'No matching exported row is not evidence of idle hardware. MC2 internal computation or communication remains opaque.',
       'Each calculation uses a single rank CSV clock; summaries use distributions of local ratios, no cross-rank timestamp matching.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
