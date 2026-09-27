#!/usr/bin/env python3
"""V3.36: Run580 copied Level0 timeline attribution for mixed service."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from extreme_bound_calibration_v3_35 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT=Path(__file__).resolve().parents[1]
PRIOR='evidence/20260928_loop080_bound/run579/bound_calibration_v3_35.json'
PRIOR_SHA='410763b5a05e5cc06af9d2d303e60481a29ab67f64a36ba2c4a04c4e773bea63'
FINAL='evidence/20260928_loop080_bound/run580/final_admission.json'
FINAL_SHA='e6201f16232f3d629d81f47e1e0cc81ef97139eee41c208a2f37c4eb561959b4'
PARSE='evidence/20260928_loop080_bound/run580/offline_parse/manifest.json'
PARSE_SHA='9ca7cb832230284803eff1d5977c6af0a492b69b7532096979af2941d166386c'
TIMELINE='evidence/20260928_loop080_bound/run580/timeline_attribution.json'
TIMELINE_SHA='1954e4508a7422a83a4e11767b5539bfb3b08960ce76a656cb52fb1f22cf36e4'
REDUCER='scripts/loop080_mixed_timeline_reduce_run580.py'
REDUCER_SHA='ed67d18a7ba94bedc18aa22b980e285d2fab2609c099ed2f165ebff69b9a2b09'
PARSER='scripts/loop080_mixed_offline_parse_run580.py'
PARSER_SHA='6362b6156f2aae517c36450c5cb999bc2c822f803eaf0856545f1056c5ee86d7'

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def build():
    need(sha(PRIOR)==PRIOR_SHA,'V3.35 prior SHA drift')
    prior=build_prior()
    need(json.dumps(prior,ensure_ascii=False,indent=2)+'\n'==(ROOT/PRIOR).read_text(),
         'V3.35 generator/output drift')
    need(prior['model_revision']=='V3.35' and
         prior['current']['accepted_formal_tps']==571.681,'formal Current drift')
    for path,expected in ((FINAL,FINAL_SHA),(PARSE,PARSE_SHA),(TIMELINE,TIMELINE_SHA),
                          (REDUCER,REDUCER_SHA),(PARSER,PARSER_SHA)):
        need(sha(path)==expected,f'Run580 admitted input SHA drift: {path}')
    final=json.loads((ROOT/FINAL).read_text())
    parsed=json.loads((ROOT/PARSE).read_text())
    trace=json.loads((ROOT/TIMELINE).read_text())
    need(final['status']=='scoped_terminal_attained_service_admitted' and
         final['run_tag']=='LOOP080-RUN580-B' and final['source_restore_exact'] and
         final['script_sha_exact'] and final['final_all8_idle'] and
         final['post_stop_posts']==96 and len(final['all_cohort_runtime_provenance'])==64 and
         len(final['raw_profile_provenance'])==534,
         'Run580 final admission drift')
    for section in ('service_provenance','all_cohort_runtime_provenance',
                    'post_stop_provenance','raw_profile_provenance'):
        need(section in final and final[section],f'Run580 missing {section}')
        for path,expected in final[section].items():
            need(sha(path)==expected,f'Run580 {section} drift: {path}')
    need(parsed['status']=='offline_copied_parse_complete' and
         parsed['raw_source_immutable'] and parsed['final_admission_sha256']==FINAL_SHA and
         len(parsed['parsed'])==8 and
         trace['status']=='scoped_all8_attributed_level0_timeline' and
         trace['run_tag']=='LOOP080-RUN580-B' and len(trace['rank_rows'])==8,
         'Run580 copied timeline attribution drift')
    for path,expected in trace['provenance'].items():
        need(sha(path)==expected,f'Run580 parsed timeline provenance drift: {path}')
    need(all(final[k] is None for k in ('strict_capacity_upper',
         'resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
         'product_e2e_tps_interval','numeric_current_to_limit_distance')) and
         all(trace[k] is None for k in ('strict_capacity_upper',
         'resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
         'product_e2e_tps_interval','numeric_current_to_limit_distance')),
         'Run580 numeric promotion attempted')
    rows=trace['rank_rows']
    need(all(row['gmm_model_id']==33 and row['hccl_model_id']==32 and
             row['serial']['gmm_task_count']==row['concurrent']['gmm_task_count']==86 and
             row['serial']['hccl_task_count']==row['concurrent']['hccl_task_count']==265 and
             0.3<row['concurrent']['hccl_first_minus_gmm_first_ms']<0.5 and
             row['concurrent']['native_interval_overlap_ms']>12 and
             84<=row['matched_ordinal_hccl_task_duration']['prefix_call_count']<=86 and
             row['matched_ordinal_hccl_task_duration']['tasks_crossing_gmm_end']==1
             for row in rows),'Run580 two-Graph task attribution drift')
    prior['model_revision']='V3.36'
    prior['bound_semantics_v3_36']={
        'active_objective':'minimum execution time for unchanged DSpark7 logical W0, acceptance, cycles and outputs',
        'measurement_class':'scoped_copied_level0_mechanism_timeline',
        'run580':{
            'unprofiled_four_arm_slowest_rank_median_ms':final['arm_slowest_rank_median_joint_ms'],
            'two_graph_task_identity':'all8 Model33 GMM 86 native tasks and Model32 HCCL 265 AIV tasks per active arm; barriers excluded',
            'concurrent_hccl_first_minus_gmm_first_ms':[r['concurrent']['hccl_first_minus_gmm_first_ms'] for r in rows],
            'concurrent_task_outer_interval_overlap_ms':[r['concurrent']['native_interval_overlap_ms'] for r in rows],
            'serial_hccl_task_span_ms':[r['serial']['hccl_span_ms'] for r in rows],
            'concurrent_hccl_task_span_ms':[r['concurrent']['hccl_span_ms'] for r in rows],
            'matched_ordinal_hccl_task_duration':[r['matched_ordinal_hccl_task_duration'] for r in rows],
            'claim':'late initial whole-chain HCCL Graph launch alone does not explain the slowdown; exported task intervals overlap and HCCL task-time inflation occurs during GMM task-envelope coexistence, returning near serial level after GMM ends',
            'confidence':'high for this copied all8 task identity and relative rank-local timing; low for precise HBM/AIV/link/rank-wait attribution or production transfer',
        },
        'not_proven':['task outer-interval overlap as simultaneous useful compute/communication occupancy',
                      'fraction of slowdown due HBM, AIV, HCCL internal synchronization, scheduler or rank arrival',
                      'formal Run99 W0 or production dependency-legal overlap',
                      'compulsory work/traffic and exact-board cumulative C+/B',
                      'finite Resource/Scheduling/Product endpoint or numeric Current-to-limit distance'],
        'resource_hardware_endpoint_s':None,
        'scheduling_execution_endpoint_s':None,
        'product_e2e_tps_interval':None,
        'numeric_distance_from_current_tps':None,
        'next_bound_measurements':[
            'Trace the actual production producer-ready, collective submit/completion, consumer-ready, Target/Draft/KV/Host edges for unchanged W0, then solve a dependency-aware critical-path lower bound and legal overlap windows.',
            'Measure necessary W-minus/traffic and exact-board cumulative C+/B with matched shapes/layout; use targeted HCCL arrival or Level1 PMU only if it changes a legal scheduling decision.',
            'Test shorter dependency-preserving overlap segments only after DAG and resource feasibility are established.'
        ],
    }
    prior['next_measurement']['priority']=prior['bound_semantics_v3_36']['next_bound_measurements'][0]
    prior['input_paths'] += [PRIOR,FINAL,PARSE,TIMELINE,REDUCER,PARSER]
    require_null_endpoints(prior)
    need(all(prior['bound_semantics_v3_36'][k] is None for k in
             ('resource_hardware_endpoint_s','scheduling_execution_endpoint_s',
              'product_e2e_tps_interval','numeric_distance_from_current_tps')),
         'V3.36 unproved endpoint filled')
    need(not any(prior['proof_dag']['certified'].values()),'unproved certificate promoted')
    return prior

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=build();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'ok','revision':'V3.36','finite_endpoints':0}))
if __name__=='__main__':main()
