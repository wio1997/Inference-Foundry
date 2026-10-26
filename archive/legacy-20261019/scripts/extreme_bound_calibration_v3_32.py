#!/usr/bin/env python3
"""V3.32: fixed-W0 Bound ledger with Run576 conditional operand census."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_31 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints


ROOT=Path(__file__).resolve().parents[1]
PRIOR='evidence/20260927_loop080_bound/run560/bound_calibration_v3_31.json'
PRIOR_SHA='d425151424e9ca42580c04b022f26d1bf09fd2fa3d0d44e58eb585040a26f6ab'
CENSUS='evidence/20260928_loop080_bound/run576/operand_census.json'
CENSUS_SHA='615aeb42086b035903867261e45642a6390a13ba396d91ec31d7cb62320ca1e8'
JOIN='evidence/20260928_loop080_bound/run576/live/b/joined_admission.json'
CAPTURE='evidence/20260928_loop080_bound/run576/live/b/capture_admission.json'


def need(ok,why):
    if not ok:raise ValueError(why)


def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def build():
    need(sha(PRIOR)==PRIOR_SHA,'V3.31 prior SHA drift')
    model=build_prior()
    need(json.dumps(model,ensure_ascii=False,indent=2)+'\n' ==
         (ROOT/PRIOR).read_text(),'V3.31 generator/output mismatch')
    need(model['model_revision']=='V3.31' and
         model['current']['accepted_formal_tps']==571.681,'prior/formal Current drift')
    need(sha(CENSUS)==CENSUS_SHA,'Run576 operand census SHA drift')
    census=json.loads((ROOT/CENSUS).read_text())
    need(sha(JOIN)==census['provenance'].get(JOIN) and
         sha(CAPTURE)==census['provenance'].get(CAPTURE),
         'Run576 joined/capture admission SHA drift')
    joined=json.loads((ROOT/JOIN).read_text())
    capture=json.loads((ROOT/CAPTURE).read_text())
    need(census['status']=='scoped_conditional_operand_census' and
         joined['status']=='diagnostic_capture_client_runtime_join_pass' and
         capture['status']=='scoped_diagnostic_accepted','Run576 admission')
    need(census['selected_cycle']==joined['cycle']==capture['cycle']==170 and
         census['total_cycles']==capture['total_cycles']==286,'Run576 cycle identity')
    need(census['models']['target']['global_route_incidence_per_cycle']==24768 and
         census['models']['draft']['global_route_incidence_per_cycle']==1512,
         'Run576 route incidence drift')
    need(all(census['models'][name]['all8_owned_incidence']==
             census['models'][name]['global_route_incidence_per_cycle']
             for name in ('target','draft')),'all8 owner conservation')
    need(census['formal_W0_transfer'] is False and
         census['latency_or_tps_inference'] is False and
         census['algorithm_configuration_edit'] is False and
         census['trajectory_noninterference_proven'] is False and
         census['strict_numeric_bounds_unchanged'] is True,
         'diagnostic promotion attempted')
    model['model_revision']='V3.32'
    model['bound_semantics_v3_32']={
        'active_objective':'minimize execution time for fixed DSpark7 W0, acceptance, cycles and output semantics',
        'run576_scope':'one admitted measured cohort5 cycle170 after natural slot1 park; route and entry operand observation',
        'run576_target':{
            'physical_rows_per_layer':96,'routed_layers':43,'topk':6,
            'global_route_incidence':24768,
            'all8_nonempty_local_expert_layer_pairs':census['models']['target']['all8_nonempty_local_expert_layer_pairs'],
            'selected_expert_slice_arithmetic_bytes_per_rank':[
                x['selected_expert_slice_arithmetic_bytes'] for x in census['models']['target']['per_rank']],
            'resident_operand_storage_bytes_per_rank':[
                x['unique_operand_storage_bytes'] for x in census['models']['target']['per_rank']],
        },
        'run576_dspark':{
            'physical_rows_per_layer':84,'routed_layers':3,'topk':6,
            'global_route_incidence':1512,
            'selected_expert_slice_arithmetic_bytes_per_rank':[
                x['selected_expert_slice_arithmetic_bytes'] for x in census['models']['draft']['per_rank']],
            'resident_operand_storage_bytes_per_rank':[
                x['unique_operand_storage_bytes'] for x in census['models']['draft']['per_rank']],
        },
        'measurement_class':'diagnostic_conditional_ownership_and_operand_geometry',
        'confidence':'high for observed post-park rank/route/group/operand metadata and join; low for physical selected-slice read semantics or formal transfer; trajectory noninterference unproven',
        'not_proven':['formal Run99 W0 work vector','43-layer semantic row composition',
                      'actual native replay dynamic argument lineage','compulsory HBM traffic',
                      'exact-board cumulative C+/B','all8 mixed attainable service',
                      'legal Product schedule and numeric Current-to-limit distance'],
        'resource_hardware_endpoint_s':None,
        'scheduling_execution_endpoint_s':None,
        'product_e2e_tps_interval':None,
        'numeric_distance_from_current_tps':None,
        'next_bound_measurements':[
            'Measure bounded all8 real-weight GMM Graph service with Run576-matched route/group distribution and cross-layer working set; distinguish attainable Engineering capacity from synthetic-zero or isolated service and do not treat it as C+.',
            'Pair one real all8 fixed-W0 Target+DSpark mixed-service/counter window with admitted route and operand provenance; distinguish selected packed payload from actual GMM/HBM read and concurrent compute/HCCL service.',
            'Obtain formal-window necessary-work witness and matching exact-board cumulative C+/B certificate independently; do not promote attained isolated rates to strict C+.',
        ],
    }
    model['next_measurement']['priority']=model['bound_semantics_v3_32']['next_bound_measurements'][0]
    model['input_paths'] += [PRIOR,CENSUS,JOIN,CAPTURE]
    require_null_endpoints(model)
    need(not any(model['proof_dag']['certified'].values()),'unproved certificate promoted')
    v=model['bound_semantics_v3_32']
    need(all(v[k] is None for k in ('resource_hardware_endpoint_s',
         'scheduling_execution_endpoint_s','product_e2e_tps_interval',
         'numeric_distance_from_current_tps')),'new unproved endpoint filled')
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    model=build()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'ok','revision':'V3.32','finite_endpoints':0}))


if __name__=='__main__':main()
