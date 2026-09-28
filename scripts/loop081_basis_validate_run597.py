#!/usr/bin/env python3
"""Fail-closed all8 Run597 diagnostic basis/Runtime/client join."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import loop081_basis_capture_run597 as capture

ROOT=Path('/data/wio/Inference_Foundry')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def semantic(record):
    model=[]
    for row in record['draft_model_witnesses']:
        model.append({k:v for k,v in row.items()
                      if k not in ('group_query_slots','group_context_slots')})
    return dict(req_ids=record['req_ids'],cycles=record['cycles'],
        initial=record['initial'],token_history=record['token_history'],
        count_history=record['count_history'],draft_history=record['draft_history'],
        host_park_events=record['host_park_events'],
        branch_history=record['branch_history'],
        target_witnesses=record['target_witnesses'],
        dspark_witnesses=record['dspark_witnesses'],
        draft_model_witnesses=model,
        generated_output_counts=record['generated_output_counts'],
        staged_output_counts=record['staged_output_counts'])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm-dir',required=True,type=Path)
    ap.add_argument('--run-ts',required=True)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    need(not args.output.exists(),'output already exists')
    basis_dir=args.arm_dir/'basis';runtime_dir=args.arm_dir/'runtime'
    need(len(list(basis_dir.glob('*.json')))==32,'exact32 basis files')
    need(len(list(runtime_dir.glob('*.json')))==64,'exact64 Runtime files')
    client_path=args.arm_dir/'client_admission.json'
    client=json.loads(client_path.read_text())
    need(client['status']=='client_two_phase_admitted' and
         client['request_count']==96 and client['unique_response_ids']==96 and
         client['same_48_request_bodies_across_phases'] is True and
         len(client['request_index'])==96,'client admission')
    by_response={x['response_id']:x for x in client['request_index']}
    need(len(by_response)==96,'client response uniqueness')
    posts=int((args.arm_dir/'server_post_count.txt').read_text().strip())
    need(posts==96,'server POST count')
    hashes={str(client_path.relative_to(ROOT)):digest(client_path)}
    rows=[];refs={};rank0={};client_seen=set();client_keys={}
    for rank in range(8):
        for cohort in range(1,9):
            rp=runtime_dir/f'rank{rank}_cohort{cohort}.json'
            runtime=json.loads(rp.read_text())
            hashes[str(rp.relative_to(ROOT))]=digest(rp)
            need(runtime['rank']==rank and runtime['cohort']==cohort and
                 runtime['pass'] and len(runtime['req_ids'])==12 and
                 len(set(runtime['req_ids']))==12 and
                 runtime['generated_output_counts']==[1024]*12,
                 f'Runtime identity/output rank{rank} cohort{cohort}')
            if cohort not in refs:refs[cohort]=dict(req_ids=runtime['req_ids'],
                cycles=runtime['cycles'],staged=runtime['staged_output_counts'],
                generated=runtime['generated_output_counts'])
            else:
                ref=refs[cohort]
                need((runtime['req_ids']==ref['req_ids'] and
                      runtime['cycles']==ref['cycles'] and
                      runtime['staged_output_counts']==ref['staged'] and
                      runtime['generated_output_counts']==ref['generated']),
                     f'all8 Runtime disagreement cohort{cohort}')
            if rank==0:
                keys=[]
                for request_id in runtime['req_ids']:
                    matches=[item for response_id,item in by_response.items()
                             if request_id.startswith(response_id+'-')]
                    need(len(matches)==1 and request_id not in client_seen,
                         f'client/worker request identity cohort{cohort}')
                    client_seen.add(request_id)
                    item=matches[0]
                    need(item['phase']==('warmup' if cohort<=4 else 'measured'),
                         f'client phase cohort{cohort}')
                    keys.append(dict(dataset_index=item['dataset_index'],
                        dataset_row_sha256=item['dataset_row_sha256'],
                        request_body_sha256=item['request_body_sha256'],
                        response_id=item['response_id']))
                client_keys[str(cohort)]=keys
            if cohort<5:continue
            bp=basis_dir/f'rank{rank}_cohort{cohort}.json'
            record=json.loads(bp.read_text())
            hashes[str(bp.relative_to(ROOT))]=digest(bp)
            need(record['rank']==rank and record['cohort']==cohort and
                 record['run_ts']==args.run_ts and
                 record['req_ids']==runtime['req_ids'] and
                 record['cycles']==runtime['cycles'] and
                 record['staged_output_counts']==runtime['staged_output_counts'] and
                 record['generated_output_counts']==runtime['generated_output_counts'],
                 f'basis/Runtime identity rank{rank} cohort{cohort}')
            check=capture._check_basis(record)
            need(check==record['basis_check'],f'basis self-check rank{rank} cohort{cohort}')
            semantic_sha=hashlib.sha256(json.dumps(semantic(record),
                sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if cohort in rank0:
                need(semantic_sha==rank0[cohort],f'all8 semantic basis cohort{cohort}')
            else:rank0[cohort]=semantic_sha
            rows.append(dict(rank=rank,cohort=cohort,cycles=record['cycles'],
                active_target8_rows=check['active_target8_rows'],
                current_physical_target_rows=check['current_physical_target_rows'],
                target_checkpoint_cycles=check['target_checkpoint_cycles'],
                dspark_checkpoint_cycles=check['dspark_checkpoint_cycles'],
                semantic_sha256=semantic_sha))
    need(len(rows)==32 and len(refs)==8 and len(rank0)==4 and
         len(client_seen)==96,'coverage')
    for cohorts in (range(1,5),range(5,9)):
        indices=[item['dataset_index'] for c in cohorts
                 for item in client_keys[str(c)]]
        need(sorted(indices)==list(range(48)),'client dataset 0..47 per phase')
    totals=dict(cycles=sum(refs[c]['cycles'] for c in range(5,9)),
        active_target8_rows=sum(r['active_target8_rows'] for r in rows if r['rank']==0),
        current_physical_target_rows=sum(r['current_physical_target_rows']
                                         for r in rows if r['rank']==0))
    need(totals['current_physical_target_rows']==96*totals['cycles'],
         'physical cycle geometry')
    result=dict(status='diagnostic_compact_basis_all8_admitted',
        scope='new measured W0; collector perturbing; Runtime role ledger only',
        run_ts=args.run_ts,rows=rows,measured_rank0_totals=totals,
        source_sha256=hashes,client_cohort_join=client_keys,
        all8_semantic_basis_consensus=True,
        formal_Run99_same_state=False,full_draft_kv_prefill_work=False,
        strict_resource_bound_s=None,strict_scheduling_bound_s=None,
        strict_product_bound_tps=None,formal_current_tps=571.681)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'measured_rank0_totals':totals,
                      'basis_files':32,'runtime_files':64}))

if __name__=='__main__':main()
