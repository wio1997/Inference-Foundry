#!/usr/bin/env python3
"""Post-stop admission for Run578 isolated Engineering service evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from loop080_realweight_reduce_run578 import ROOT, build, need


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    live=ROOT/'evidence/20260928_loop080_bound/run578/live'
    arm=live/'b'
    admission=arm/'service_admission.json'
    expected=json.dumps(build(arm),indent=2)+'\n'
    need(admission.read_text()==expected,'Run578 service reducer/output mismatch')
    cleanup=dict(line.split('=',1) for line in (live/'cleanup_status.txt').read_text().splitlines())
    required=('run_exit','stop_exit','stop_verify_exit','restore_exit',
              'source_sha_exit','source_compare_exit','script_sha_exit',
              'script_compare_exit','final_exit')
    need(all(cleanup.get(key)=='0' for key in required),'Run578 controller cleanup gate')
    need((live/'source_before.sha256').read_bytes()==(live/'source_after.sha256').read_bytes(),
         'Run578 borrowed source restore SHA')
    need((live/'scripts_before.sha256').read_bytes()==(live/'scripts_after.sha256').read_bytes(),
         'Run578 diagnostic script SHA drift')
    need(int((arm/'server_post_count.txt').read_text())==
         int((arm/'server_post_count_after_stop.txt').read_text())==96,
         'Run578 post-stop POST count')
    stop_log=(live/'final_stop_verify.log').read_text()
    need(stop_log.strip().splitlines()[-1].startswith('idle HBM [') and
         (live/'stop_npu_probe.txt').read_text().count('No running processes found in NPU')==8,
         'Run578 all8 final idle')
    report=json.loads(expected)
    client=json.loads((arm/'client_admission.json').read_text())
    runtime_paths=sorted((arm/'runtime').glob('rank*_cohort*.json'))
    need(len(runtime_paths)==64,'Run578 all-cohort Runtime file count')
    all_runtime_provenance={}
    for cohort in range(1,9):
        request_ids=None
        for rank in range(8):
            path=arm/'runtime'/f'rank{rank}_cohort{cohort}.json'
            raw=path.read_bytes()
            all_runtime_provenance[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
            row=json.loads(raw)
            need(row['rank']==rank and row['cohort']==cohort and row['pass'] is True and
                 row['host_mirror_exact'] is True and row['generated_output_counts']==[1024]*12 and
                 row['target_graph_requested'] is True and row['target_graph_mode']=='FULL',
                 f'Run578 rank{rank} cohort{cohort} Runtime contract')
            if request_ids is None: request_ids=row['req_ids']
            need(row['req_ids']==request_ids,f'Run578 cohort{cohort} all8 request parity')
        need(len(request_ids)==len(set(request_ids))==12,'Run578 cohort request uniqueness')
        matched=[]
        for req in request_ids:
            options=[item for item in client['request_index']
                     if req==item['response_id'] or req.startswith(item['response_id']+'-')]
            need(len(options)==1,f'Run578 cohort{cohort} client ID join')
            matched.append(options[0])
        phase='warmup' if cohort<=4 else 'measured'
        indices=set(range(((cohort-1)%4)*12,((cohort-1)%4+1)*12))
        need({item['phase'] for item in matched}=={phase} and
             {item['dataset_index'] for item in matched}==indices,
             f'Run578 cohort{cohort} client dataset set')
    provenance={str(path.relative_to(ROOT)):sha(path) for path in
        (admission,live/'cleanup_status.txt',live/'source_before.sha256',
         live/'source_after.sha256',live/'scripts_before.sha256',
         live/'scripts_after.sha256',live/'final_stop_verify.log',
         live/'stop_npu_probe.txt',arm/'server_post_count_after_stop.txt')}
    final=dict(status='scoped_terminal_attained_service_admitted',
               run_tag='LOOP080-RUN578-B',
               measured_slowest_rank_median_ms=report['slowest_rank_median_ms'],
               rank_median_ms=report['rank_median_ms'],
               rank_median_after_ms=report['rank_median_after_ms'],
               profile_per_rank=report['profile_per_rank'],
               counter_semantics=report['counter_semantics'],
               all8_host_window_overlap_ms=report['all8_host_window_overlap_ms'],
               service_provenance=report['provenance'],
               all_cohort_runtime_provenance=all_runtime_provenance,
               post_stop_provenance=provenance,
               source_restore_exact=True,script_sha_exact=True,
               final_all8_idle=True,post_stop_posts=96,
               classification='attained_isolated_engineering_service_only',
               strict_capacity_upper=None,
               resource_hardware_endpoint_s=None,
               scheduling_execution_endpoint_s=None,
               product_e2e_tps_interval=None,
               numeric_current_to_limit_distance=None,
               limits=report['limits']+[
                   'Production W4A8 weights are real; activation and scale are private synthetic inputs.',
                   '43-layer GMM chain is isolated after final cohort and does not include Draft, HCCL, KV or Product schedule.',
                   'Process-cumulative memory peaks are not benchmark incremental allocations.',
                   'Profiler main-memory counters require current CANN definition before any physical HBM or attainable-bandwidth inference.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({'status':final['status'],
                      'slowest_rank_median_ms':final['measured_slowest_rank_median_ms']}))


if __name__=='__main__':main()
