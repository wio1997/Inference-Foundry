#!/usr/bin/env python3
"""V3.35: Run579 corrected independent-ready GMM × TP8 HCCL service."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
from extreme_bound_calibration_v3_34 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT=Path(__file__).resolve().parents[1]
PRIOR='evidence/20260928_loop080_bound/run578/bound_calibration_v3_34.json'
PRIOR_SHA='7ee7eac00ece06cbcf13d9e18d6c023cb75d09cfb0616804629c4a3e147749df'
RUN='evidence/20260928_loop080_bound/run579/final_admission.json'
RUN_SHA='64233da44fcb4051c6cc067ed21c7e37bd0d2389155e4c5e9d46ba7f9911e45e'
FINALIZER='scripts/loop080_mixed_final_run579.py'
FINALIZER_SHA='90eeb1a3f0aa2235b4f8e68dfb7d33b133a08ed1243d8f80bf03f4b52860482f'
REDUCER='scripts/loop080_mixed_reduce_run579.py'
REDUCER_SHA='452b89e2db5e3724de93818e87b33cdcbee8bbb2b8f63ac1de222b99a6dda8ee'
GATE='scripts/loop080_mixed_reducer_gate_run579.py'
GATE_SHA='24b5564004739e90eed65e7f698736ab1ba3eb93f705c584c7733caeaea0812a'
CHAIN='scripts/loop080_mixed_chain_run579.py'
CHAIN_SHA='ecee9e41aba0cd73d4d0e5430c2e72ccf3f770cf8b34f77ea7209b7098eea3db'
SERVICE='scripts/loop080_mixed_service_run579.py'
SERVICE_SHA='7307668fc7ad73e5445cfb7184117cd3746c684551c74f073f6fdd619188ceeb'

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def build():
    need(sha(PRIOR)==PRIOR_SHA,'V3.34 prior SHA drift')
    prior=build_prior()
    need(json.dumps(prior,ensure_ascii=False,indent=2)+'\n'==(ROOT/PRIOR).read_text(),
         'V3.34 generator/output drift')
    need(prior['model_revision']=='V3.34' and
         prior['current']['accepted_formal_tps']==571.681,'formal Current drift')
    for path,expected in ((RUN,RUN_SHA),(FINALIZER,FINALIZER_SHA),
                          (REDUCER,REDUCER_SHA),(GATE,GATE_SHA),
                          (CHAIN,CHAIN_SHA),(SERVICE,SERVICE_SHA)):
        need(sha(path)==expected,f'Run579 report/code SHA drift: {path}')
    run=json.loads((ROOT/RUN).read_text())
    for section in ('service_provenance','all_cohort_runtime_provenance','post_stop_provenance'):
        need(section in run and run[section],f'Run579 missing provenance {section}')
        for path,expected in run[section].items():
            need(sha(path)==expected,f'Run579 raw provenance drift: {path}')
    need(run['status']=='scoped_terminal_attained_service_admitted' and
         run['run_tag']=='LOOP080-RUN579-B' and
         run['source_restore_exact'] and run['script_sha_exact'] and
         run['final_all8_idle'] and run['post_stop_posts']==96 and
         len(run['all_cohort_runtime_provenance'])==64 and
         run['classification']=='attained_isolated_engineering_service_only',
         'Run579 final admission drift')
    need(all(run[k] is None for k in ('strict_capacity_upper',
         'resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
         'product_e2e_tps_interval','numeric_current_to_limit_distance')),
         'Run579 numeric promotion attempted')
    arms=run['arm_rank_median_joint_ms']
    need(set(arms)=={'gmm','hccl','serial','concurrent'} and
         all(len(v)==8 and all(0<x<1000 for x in v) for v in arms.values()),
         'Run579 four-arm shape/value drift')
    deltas=[c-s for c,s in zip(arms['concurrent'],arms['serial'])]
    need(all(1.70<x<1.85 for x in deltas) and
         abs(max(arms['concurrent'])-16.204840660095215)<1e-6 and
         abs(max(arms['serial'])-14.40155029296875)<1e-6,
         'Run579 measured contrast drift')
    prior['model_revision']='V3.35'
    prior['bound_semantics_v3_35']={
        'active_objective':'minimum execution time for unchanged DSpark7 logical W0, acceptance, cycles and outputs',
        'measurement_class':'scoped_same_run_independent_ready_mixed_service',
        'run579':{
            'four_arm_rank_local_joint_median_ms':arms,
            'slowest_rank_median_joint_ms':run['arm_slowest_rank_median_joint_ms'],
            'concurrent_minus_serial_rank_median_ms':deltas,
            'correctness':'all8 three fresh HCCL generations and fourth common-start fresh dual-Graph validation; all265 outputs and GMM valid rows',
            'weights':'loaded production W4A8 Target resident operands',
            'routes':'Run576 cohort5 cycle170 group fixture',
            'inputs':'private synthetic nonzero GMM activation/scale and independent pre-ready per-collective communication buffers',
            'claim':'whole-chain independent-ready dual Graph concurrent service was slower than same-run serial service on every rank; naive max-of-isolated-times overlap assumption is unsupported',
            'confidence':'high for this correctly admitted same-run conditional protocol; low transfer to production DAG or alternative segmented schedules',
        },
        'not_proven':['formal Run99 W0 equivalence or Product E2E gain',
                      'production producer/consumer dependency-legal overlap',
                      'exact cause of concurrent interference',
                      'compulsory compute/traffic/communication or exact-board cumulative C+/B',
                      'finite Resource, Scheduling or Product endpoint and Current-to-limit distance'],
        'resource_hardware_endpoint_s':None,
        'scheduling_execution_endpoint_s':None,
        'product_e2e_tps_interval':None,
        'numeric_distance_from_current_tps':None,
        'next_bound_measurements':[
            'Acquire a short-window all8 serial/concurrent task timeline for the same Run579 configuration and parse it offline; distinguish delayed HCCL launch, interleaved GMM service inflation and collective rank-arrival wait. Use Run579 unprofiled timings as the service reference.',
            'Record actual production input-ready, submit, completion and consumer-ready edges for ordered collectives, Target, DSpark, KV and Host on an unchanged W0 trajectory; identify legal overlap windows in the critical-path DAG.',
            'Where legal overlap exists, test dependency-preserving shorter segments with common-start all8 correctness; the Run579 whole-chain concurrent schedule is a counterexample to simple max-of-isolated-times composition.',
            'Certify formal necessary work/traffic W-minus and exact-board cumulative C+/B before deriving strict Resource or Product endpoints.'
        ],
    }
    prior['next_measurement']['priority']=prior['bound_semantics_v3_35']['next_bound_measurements'][0]
    prior['input_paths'] += [PRIOR,RUN,FINALIZER,REDUCER,GATE,CHAIN,SERVICE]
    require_null_endpoints(prior)
    need(all(prior['bound_semantics_v3_35'][k] is None for k in
             ('resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
              'product_e2e_tps_interval','numeric_distance_from_current_tps')),
         'V3.35 unproved endpoint filled')
    need(not any(prior['proof_dag']['certified'].values()),'unproved certificate promoted')
    return prior

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=build();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'ok','revision':'V3.35','finite_endpoints':0}))
if __name__=='__main__':main()
