#!/usr/bin/env python3
"""V3.33: fixed-W0 ledger plus Run577 attained isolated real-weight GMM service."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_32 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT=Path(__file__).resolve().parents[1]
PRIOR='evidence/20260928_loop080_bound/run576/bound_calibration_v3_32.json'
PRIOR_SHA='5e047fcd421b1a82de1396c97b55ffeea2aa4cc7d292790a278731af2ffd2a8a'
RUN='evidence/20260928_loop080_bound/run577/final_admission.json'
RUN_SHA='6699eabbaa9efab9cb3a803153144e2b25d161adf4ddc92947b1b9a4a063e21c'


def need(ok,why):
    if not ok: raise ValueError(why)


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def build():
    need(sha(PRIOR)==PRIOR_SHA,'V3.32 prior SHA drift')
    model=build_prior()
    need(json.dumps(model,ensure_ascii=False,indent=2)+'\n'==(ROOT/PRIOR).read_text(),
         'V3.32 generator/output mismatch')
    need(model['model_revision']=='V3.32' and
         model['current']['accepted_formal_tps']==571.681,
         'V3.32/formal Current drift')
    need(sha(RUN)==RUN_SHA,'Run577 final admission SHA drift')
    run=json.loads((ROOT/RUN).read_text())
    for path, expected in {**run['service_provenance'],
                           **run['all_cohort_runtime_provenance'],
                           **run['post_stop_provenance']}.items():
        need(sha(path)==expected,f'Run577 raw provenance drift: {path}')
    need(run['status']=='scoped_terminal_attained_service_admitted' and
         run['classification']=='attained_isolated_engineering_service_only' and
         run['run_tag']=='LOOP080-RUN577-B' and
         run['source_restore_exact'] and run['script_sha_exact'] and
         run['final_all8_idle'] and run['post_stop_posts']==96,
         'Run577 terminal admission')
    need(len(run['rank_median_ms'])==8 and
         abs(max(run['rank_median_ms'])-10.13116979598999)<1e-9 and
         abs(run['measured_slowest_rank_median_ms']-max(run['rank_median_ms']))<1e-9,
         'Run577 service number drift')
    need(all(run[k] is None for k in ('strict_capacity_upper','resource_hardware_endpoint_s',
         'scheduling_execution_endpoint_s','product_e2e_tps_interval',
         'numeric_current_to_limit_distance')),'Run577 numeric promotion attempted')
    model['model_revision']='V3.33'
    model['bound_semantics_v3_33']={
        'active_objective':'minimum execution time for unchanged DSpark7 logical W0, acceptance, cycles and outputs',
        'measurement_class':'attained_isolated_engineering_service_prior',
        'run577_real_weight_target_gmm_graph':{
            'layers':43,'gmm_pairs_per_replay':43,'ranks':8,
            'samples_per_rank':20,
            'rank_median_ms':run['rank_median_ms'],
            'slowest_rank_median_ms':run['measured_slowest_rank_median_ms'],
            'all8_host_window_overlap_ms':run['all8_host_window_overlap_ms'],
            'route_group_source':'Run576 cohort5 cycle170, SHA-pinned census/provenance',
            'weight_source':'actual loaded production W4A8 Target weights in terminal diagnostic',
            'activation_source':'private deterministic nonzero synthetic input/scale',
            'execution_scope':'43 GMM1→GMM2 pairs, private Graph, no other model/collective work',
            'confidence':'high for scoped reproduced service and all8 Host window; low for Product/mixed-resource transfer',
        },
        'not_proven':['formal Run99 W0 GMM calls/routes/activation',
                      'strict cumulative compute/HBM capacity C+/B',
                      'actual GMM HBM bytes under this fixture',
                      'cross-rank device clock/physical overlap',
                      'mixed Target+DSpark+HCCL/KV attainable service',
                      'legal fixed-W0 Scheduling critical path',
                      'Product E2E TPS ceiling or numeric Current-to-limit distance'],
        'resource_hardware_endpoint_s':None,
        'scheduling_execution_endpoint_s':None,
        'product_e2e_tps_interval':None,
        'numeric_distance_from_current_tps':None,
        'next_bound_measurements':[
            'Measure GMM physical HBM traffic and task intervals for the same real-weight route/group/cross-layer fixture; keep counters and event time in one admitted all8 trial.',
            'Measure same-W0 mixed Target+DSpark+HCCL service and collective producer/arrival/completion dependence; test whether isolated GMM capability is attainable under resource contention.',
            'Independently certify formal necessary-work/traffic W-minus and exact-board cumulative C+/B before any strict Resource floor.'
        ],
    }
    model['next_measurement']['priority']=model['bound_semantics_v3_33']['next_bound_measurements'][0]
    model['input_paths'] += [PRIOR,RUN]
    require_null_endpoints(model)
    need(not any(model['proof_dag']['certified'].values()),'unproved certificate promoted')
    need(all(model['bound_semantics_v3_33'][k] is None for k in
             ('resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
              'product_e2e_tps_interval','numeric_distance_from_current_tps')),
         'V3.33 unproved endpoint filled')
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    model=build()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'ok','revision':'V3.33','finite_endpoints':0}))


if __name__=='__main__':main()
