#!/usr/bin/env python3
"""Audit a scoped Run502 static event order without promoting it to Run583 lineage."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
DUMP = ROOT / 'evidence/20260927_loop079_identity/run502/b_candidate/graph_dump'
CAPTURE = ROOT / 'evidence/20260928_loop081_bound/run583/live/b/capture'
CHAIN = [(1, 40, 'MatMulV2_ND_ND_FP16_FP16_false_true_all_99010'),
         (1, 41, 'EVENT_RECORD_'), (0, 12, 'EVENT_WAIT_'),
         (0, 13, 'aiv_reduce_scatter_bfloat16_t'),
         (0, 15, 'EVENT_RECORD_'), (1, 42, 'EVENT_WAIT_'),
         (1, 43, 'MEMCPY_ASYNC'),
         (1, 44, 'HcPost_d2fe6614722b27afa6f8d964c215447e_0')]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit(rows: list[dict], meta: dict, rank: int) -> dict:
    if (len(rows) != 5412 or meta['rank'] != rank or meta['cohort'] != 5 or
        meta['node_count'] != len(rows) or meta['native_model_ids'] != [45] or
        meta['capture_generation'] != 1 or
        meta['batch_descriptor'] !=
        'BatchDescriptor(num_tokens=96, num_reqs=12, uniform=True, has_lora=False, num_active_loras=0)'):
        raise ValueError('Run502 selected graph scope')
    ix = {}
    for row in rows:
        args = row['args']
        key = (args['Model Id'], args['Stream Id'], args['Task Id'])
        if key in ix or key[0] != 45:
            raise ValueError('duplicate task or unexpected native model')
        ix[key] = row
    chain = []
    for stream, task, name in CHAIN:
        row = ix[(45, stream, task)]
        actual = row['name']
        if name.endswith('_'):
            if not re.fullmatch(name + r'\d+', actual):
                raise ValueError('event name/type mismatch')
        elif actual != name:
            raise ValueError('native task mismatch')
        chain.append({'model': 45, 'stream': stream, 'task': task,
                      'name': actual, 'type': row['args']['Task Type']})
    if chain[1]['name'].split('_')[-1] != chain[2]['name'].split('_')[-1]:
        raise ValueError('producer event pair')
    if chain[4]['name'].split('_')[-1] != chain[5]['name'].split('_')[-1]:
        raise ValueError('collective event pair')
    if [x['type'] for x in chain] != [
        'KERNEL_AICORE', 'EVENT_RECORD', 'EVENT_WAIT', 'KERNEL_AIVEC',
        'EVENT_RECORD', 'EVENT_WAIT', 'MEMCPY_ASYNC', 'KERNEL_AIVEC']:
        raise ValueError('native task types')
    # Same-stream task order plus two cross-stream event edges; dump ts/dur
    # are synthetic and are deliberately never read here.
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]
    return {'rank': rank, 'native_model_id': 45, 'chain': chain,
            'event_ids': [chain[1]['name'].split('_')[-1],
                          chain[4]['name'].split('_')[-1]],
            'static_order_edges': edges}


def selftest(rows: list[dict], meta: dict, rank: int) -> int:
    tested = 0
    def reject(changed_rows, changed_meta):
        nonlocal tested
        try:
            admit(changed_rows, changed_meta, rank)
        except (ValueError, KeyError, TypeError):
            tested += 1
            return
        raise AssertionError('negative case unexpectedly admitted')
    m = dict(meta); m['capture_generation'] = 2; reject(rows, m)
    m = dict(meta); m['cohort'] = 4; reject(rows, m)
    r = copy.deepcopy(rows)
    next(x for x in r if x['args']['Stream Id']==0 and x['args']['Task Id']==12)['name']='EVENT_WAIT_0'
    reject(r, meta)
    r = copy.deepcopy(rows)
    next(x for x in r if x['args']['Stream Id']==0 and x['args']['Task Id']==13)['name']='wrong_rs'
    reject(r, meta)
    r = copy.deepcopy(rows); r.append(copy.deepcopy(r[0])); reject(r, meta)
    r = copy.deepcopy(rows)
    next(x for x in r if x['args']['Stream Id']==1 and x['args']['Task Id']==43)['args']['Task Type']='KERNEL_AIVEC'
    reject(r, meta)
    return tested


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    results = []
    inputs = {}
    negatives = 0
    for rank in range(8):
        stem = f'rank{rank}_cohort5_acl_graph'
        data_path = DUMP / f'{stem}.json'
        meta_path = DUMP / f'{stem}.meta.json'
        meta = json.loads(meta_path.read_text())
        if sha(data_path) != meta['sha256'] or data_path.stat().st_size != meta['bytes']:
            raise ValueError('dump SHA/size mismatch')
        rows = json.loads(data_path.read_text())
        result = admit(rows, meta, rank)
        if rank == 0:
            negatives = selftest(rows, meta, rank)
        capture_path = CAPTURE / f'rank{rank}_captures.jsonl'
        capture_rows = [json.loads(s) for s in capture_path.read_text().splitlines() if s.strip()]
        selected = [x for x in capture_rows if x.get('descriptor_fields') ==
                    {'num_tokens':96,'num_reqs':12,'uniform':True,'has_lora':False,'num_active_loras':0}]
        if len(selected)!=1:
            raise ValueError('Run583 selected capture cardinality')
        obj = selected[0]['layer0']
        pointers = {'partial':obj['sequence_rs']['partial']['data_ptr'],
                    'reduced':obj['sequence_rs']['reduced']['data_ptr'],
                    'destination':obj['attention_before']['output_destination']['data_ptr']}
        # A count across all kernel argument strings is *non-unique* by design.
        result['cross_run_pointer_occurrences_not_identity'] = {
            key:sum(f'0x{ptr:016x}' in row['args'].get('Kernel Args','') for row in rows)
            for key,ptr in pointers.items()}
        result['run583_capture_serial_not_joined_to_run502'] = selected[0]['capture_serial']
        results.append(result)
        inputs[str(data_path.relative_to(ROOT))]=sha(data_path)
        inputs[str(meta_path.relative_to(ROOT))]=sha(meta_path)
        inputs[str(capture_path.relative_to(ROOT))]=sha(capture_path)
    out = {'status':'run502_static_event_order_only',
           'scope':'Run502 Model45 FULL96 cohort5 capture generation1; separate from Run583',
           'negative_tests_rejected':negatives,
           'ranks':results,'input_sha256':inputs,
           'not_proven':['layer0 semantic operation to native task identity',
                         'Run583 tensor storage to Run502 native task identity',
                         'copy source/destination/count',
                         'native device execution time or completion',
                         'fixed-W0 Product/Scheduling/Resource endpoint']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'status':out['status'],'rank_count':len(results),
                      'negative_tests_rejected':negatives}))


if __name__ == '__main__':
    main()
