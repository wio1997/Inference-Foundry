#!/usr/bin/env python3
"""V3.34: Run578 scoped MemoryAccess counters; fixed-W0 endpoints unchanged."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
from extreme_bound_calibration_v3_33 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT=Path(__file__).resolve().parents[1]
PRIOR='evidence/20260928_loop080_bound/run577/bound_calibration_v3_33.json'
PRIOR_SHA='519e56cd37d3af8ed3c792ccf038063abcd3937af82f1d7a2cc16edd71c390df'
RUN='evidence/20260928_loop080_bound/run578/counter_salvage.json'
RUN_SHA='dd1fd626d2ef4ebb30ba86f72aae184939b1ffba6cae296dc6ad5fcb1f80e2f4'
REDUCER='scripts/loop080_profile_salvage_run578.py'
REDUCER_SHA='29ad539a68c51eb1708e92f9ec5e016119c528e9a312bdb00e75db5aec4a5d11'
GATE='evidence/20260928_loop080_bound/run578/salvage_gate.json'
GATE_SHA='981632fcf7dd1f0e92cdf2c4921174e2a1cb5ccb12f080efe15ae8ba2d74ef1d'
GATE_SCRIPT='scripts/loop080_profile_salvage_gate_run578.py'
GATE_SCRIPT_SHA='27cdcfe86e8e8ddc6b2553387ee3f5c3bcb565edf292aeab3e5ce7bc4b5328a7'

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def build():
    need(sha(PRIOR)==PRIOR_SHA,'V3.33 prior SHA drift')
    prior=build_prior()
    need(json.dumps(prior,ensure_ascii=False,indent=2)+'\n'==(ROOT/PRIOR).read_text(),
         'V3.33 generator/output drift')
    need(prior['model_revision']=='V3.33' and
         prior['current']['accepted_formal_tps']==571.681,
         'formal Current drift')
    need(sha(RUN)==RUN_SHA and sha(REDUCER)==REDUCER_SHA and
         sha(GATE)==GATE_SHA and sha(GATE_SCRIPT)==GATE_SCRIPT_SHA and
         json.loads((ROOT/GATE).read_text())==
         {'status':'cpu_only_salvage_gate_pass','negative_cases':7},
         'Run578 report/reducer/gate SHA drift')
    run=json.loads((ROOT/RUN).read_text())
    for path,expected in run['provenance'].items():
        need(sha(path)==expected,f'Run578 raw provenance drift: {path}')
    need(run['status']=='partial_counter_diagnostic_after_online_parse_failure' and
         run['run_tag']=='LOOP080-RUN578-B' and
         run['source_restore_exact'] and run['all8_idle'] and run['post_count']==96 and
         len(run['profile_per_rank'])==8 and
         all(len(row['complete_replays'])==2 and row['complete_task_count']==172
             for row in run['profile_per_rank']) and
         run['timing_status'].startswith('A0 samples were not persisted'),
         'Run578 scoped counter admission')
    need(all(run[k] is None for k in ('strict_capacity_upper',
         'resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
         'product_e2e_tps_interval','numeric_current_to_limit_distance')),
         'Run578 numeric promotion attempted')
    reads=run['rank_counter_reported_main_memory_read_GB']
    writes=run['rank_counter_reported_main_memory_write_GB']
    need(len(reads)==len(writes)==8 and
         abs(statistics.median(reads)-9.156191456)<1e-9 and
         abs(statistics.median(writes)-0.213859776)<1e-9,
         'Run578 counter drift')
    prior['model_revision']='V3.34'
    prior['bound_semantics_v3_34']={
        'active_objective':'minimum execution time for unchanged DSpark7 logical W0, acceptance, cycles and outputs',
        'measurement_class':'scoped_counter_only_after_failed_online_parse',
        'run578':{
            'counter_reported_main_memory_read_GB_per_rank_replay':reads,
            'counter_reported_main_memory_write_GB_per_rank_replay':writes,
            'median_read_GB':statistics.median(reads),
            'median_write_GB':statistics.median(writes),
            'native_tasks_per_replay':'43 GMM1 + 43 GMM2; two complete Graph replays on each of eight ranks',
            'weight_source':'loaded production W4A8 Target resident operands; terminal private diagnostic',
            'route_source':'Run576 cohort5 cycle170 group fixture, source/fixture SHA-pinned',
            'activation_source':'private deterministic nonzero synthetic activation and scale',
            'counter_semantics':run['counter_semantics'],
            'timing_status':run['timing_status'],
            'controller_status':'run/final exit1 due online parser; stop/verify/restore/source/script gates exit0',
            'confidence':'high for offline raw task identity/counter arithmetic; low for physical HBM and formal/mixed-service transfer',
        },
        'not_proven':['paired same-run A0/A1 service time or attainable bandwidth',
                      'physical HBM controller payload or compulsory traffic',
                      'formal Run99 W0 GMM work/route equivalence',
                      'exact-board cumulative compute/HBM/HCCL C+/B',
                      'mixed Target+DSpark+HCCL/KV service and legal critical-path overlap',
                      'finite Resource/Scheduling/Product bound or Current-to-limit distance'],
        'resource_hardware_endpoint_s':None,
        'scheduling_execution_endpoint_s':None,
        'product_e2e_tps_interval':None,
        'numeric_distance_from_current_tps':None,
        'next_bound_measurements':[
            'Measure fixed-work, real-weight GMM and production-sized HCCL single/serial/dual-stream joint completion with unprofiled A0/A1 and all8 correctness; this tests resource contention but not legal Product overlap.',
            'Capture the actual production collective producer/arrival/consumer dependencies and mixed Target/DSpark/KV/Host critical path for the same W0 trajectory.',
            'Certify formal necessary work/traffic W-minus and exact-board cumulative C+/B before deriving strict Resource or Product endpoints.'
        ],
    }
    prior['next_measurement']['priority']=prior['bound_semantics_v3_34']['next_bound_measurements'][0]
    prior['input_paths'] += [PRIOR,RUN,REDUCER,GATE,GATE_SCRIPT]
    require_null_endpoints(prior)
    need(all(prior['bound_semantics_v3_34'][k] is None for k in
             ('resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
              'product_e2e_tps_interval','numeric_distance_from_current_tps')),
         'V3.34 unproved endpoint filled')
    need(not any(prior['proof_dag']['certified'].values()),'unproved certificate promoted')
    return prior

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=build();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'ok','revision':'V3.34','finite_endpoints':0}))
if __name__=='__main__':main()
