#!/usr/bin/env python3
"""Post-stop service/Runtime/source admission for Run589 capture identity."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from loop081_slice_validate_run589 import select_rank

ROOT=Path('/data/wio/Inference_Foundry')
LIVE=ROOT/'evidence/20260928_loop081_bound/run589/live'
ARM=LIVE/'b'

def need(value,message):
    if not value:raise ValueError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    cleanup=dict(line.split('=',1) for line in (LIVE/'cleanup_status.txt').read_text().splitlines())
    required=('run_exit','stop_exit','stop_verify_exit','restore_exit','source_sha_exit',
              'source_compare_exit','script_sha_exit','script_compare_exit','final_exit')
    need(all(cleanup.get(x)=='0' for x in required),'Run589 controller cleanup failure')
    need((LIVE/'source_before.sha256').read_bytes()==(LIVE/'source_after.sha256').read_bytes(),
         'source restore mismatch')
    need((LIVE/'scripts_before.sha256').read_bytes()==(LIVE/'scripts_after.sha256').read_bytes(),
         'script drift')
    restore=json.loads((LIVE/'restore.json').read_text())
    need(restore['helper_drift_on_restore'] is False,'helper changed; evidence inadmissible')
    need(int((ARM/'server_post_count.txt').read_text())==96 and
         int((ARM/'server_post_count_after_stop.txt').read_text())==96,
         'POST count drift')
    need((LIVE/'stop_npu_probe.txt').read_text().count('No running processes found in NPU')==8,
         'all8 NPU idle missing')
    client=json.loads((ARM/'client_admission.json').read_text())
    dataset=Path('/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl')
    replay=ARM/'client_admission_replay.json'
    need(not replay.exists(),'stale client replay output')
    subprocess.run(['docker','exec','vllm-ascend26-dsv4f-w4a8','python3',
        str(ROOT/'scripts/loop080_frontier_phase_validate.py'),
        '--dataset',str(dataset),'--warmup-dir',str(ARM/'warmup_client'),
        '--measured-dir',str(ARM/'measured_client'),'--output',str(replay)],check=True)
    need(replay.read_bytes()==(ARM/'client_admission.json').read_bytes(),
         'client raw replay disagrees with admitted report')
    need(client['status']=='client_two_phase_admitted' and client['request_count']==96 and
         client['unique_response_ids']==96 and client['same_48_request_bodies_across_phases'],
         'client contract')
    for phase in ('warmup','measured'):
        r=client[phase+'_summary']
        need(r['n']==r['success']==48 and r['fail']==0 and
             r['concurrency']==12 and r['max_tokens']==1024,'phase contract')
    capture=json.loads((ARM/'capture_admission.json').read_text())
    need(capture['status']=='same_run_native_partial_rs_copy_identity_pass' and len(capture['rows'])==8,
         'capture branch admission')
    rebuilt=[select_rank(ARM/'capture'/f'rank{rank}_captures.jsonl',rank)
             for rank in range(8)]
    need(rebuilt==capture['rows'],'capture raw replay disagrees with admitted rows')
    runtime_hashes={}
    expected={f'rank{rank}_cohort{cohort}.json'
              for cohort in range(1,9) for rank in range(8)}
    actual={p.name for p in (ARM/'runtime').glob('rank*_cohort*.json')}
    need(actual==expected,'Runtime file set differs from exact64')
    cohort_requests={}
    for cohort in range(1,9):
        req_ids=None
        for rank in range(8):
            path=ARM/'runtime'/f'rank{rank}_cohort{cohort}.json'
            raw=path.read_bytes();runtime_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
            row=json.loads(raw)
            need(row['rank']==rank and row['cohort']==cohort and row['pass'] is True and
                 row['host_mirror_exact'] is True and row['generated_output_counts']==[1024]*12 and
                 row['target_graph_requested'] is True and row['target_graph_mode']=='FULL',
                 f'rank{rank} cohort{cohort} Runtime contract')
            if req_ids is None:req_ids=row['req_ids']
            need(row['req_ids']==req_ids,f'cohort{cohort} all8 request mismatch')
        need(len(set(req_ids))==12,f'cohort{cohort} duplicate request')
        cohort_requests[cohort]=req_ids
        matched=[]
        for req in req_ids:
            options=[item for item in client['request_index']
                     if req==item['response_id'] or req.startswith(item['response_id']+'-')]
            need(len(options)==1,'Runtime/client request join')
            matched.append(options[0])
        phase='warmup' if cohort<=4 else 'measured'
        indices=set(range(((cohort-1)%4)*12,((cohort-1)%4+1)*12))
        need({x['phase'] for x in matched}=={phase} and
             {x['dataset_index'] for x in matched}==indices,'cohort dataset/phase join')
    need(all(row['cohort']==5 and row['request_ids']==cohort_requests[5]
             for row in rebuilt),'capture→measured cohort5 Runtime/client join')
    capture_hashes={}
    expected_capture={f'rank{rank}_captures.jsonl' for rank in range(8)} | {
        f'rank{rank}_cohort5_acl_graph.json' for rank in range(8)} | {
        f'rank{rank}_cohort5_graph_meta.json' for rank in range(8)}
    need({p.name for p in (ARM/'capture').iterdir() if p.is_file()}==expected_capture,
         'capture/Graph exact24 file set')
    for rank in range(8):
        for suffix in ('captures.jsonl','cohort5_acl_graph.json','cohort5_graph_meta.json'):
            p=ARM/'capture'/f'rank{rank}_{suffix}'
            capture_hashes[str(p.relative_to(ROOT))]=sha(p)
    files=(LIVE/'cleanup_status.txt',LIVE/'source_before.sha256',LIVE/'source_after.sha256',
           LIVE/'scripts_before.sha256',LIVE/'scripts_after.sha256',LIVE/'restore.json',
           LIVE/'stop_npu_probe.txt',ARM/'client_admission.json',replay,
           ARM/'capture_admission.json')
    return dict(status='scoped_same_run_native_partial_rs_copy_identity_admitted',
                contract='fixed DSpark7 acceptance/cycle/output semantics; diagnostic only',
                current_formal_tps=571.681,
                capture_rows=capture['rows'],runtime_hashes=runtime_hashes,
                capture_hashes=capture_hashes,
                provenance={str(p.relative_to(ROOT)):sha(p) for p in files},
                native_collective_identity='same-generation event-chain RS task only',
                scoped_labeled_partial_and_copy_identity=True,
                typed_last_writer=None,
                compiled_hc_post_consumer=None,
                producer_ready_time=None,collective_completion_time=None,
                consumer_ready_time=None,scheduling_bound_s=None,
                resource_bound_s=None,product_bound_tps=None)

def main():
    data=build()
    path=ROOT/'evidence/20260928_loop081_bound/run589/final_admission.json'
    path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(status=data['status'],runtime_count=len(data['runtime_hashes']),
                          rank_count=len(data['capture_rows']))))

if __name__=='__main__':main()
