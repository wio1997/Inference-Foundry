#!/usr/bin/env python3
"""Admit clean Target FULL frontier as scoped Current evidence, not a Bound."""
import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT=Path(__file__).resolve().parents[1]
PRIOR='evidence/20260927_loop079_identity/run482/bound_calibration_v3_20.json'
INTERVALS='evidence/20260927_loop079_identity/run487/intervals.json'
VALID='evidence/20260927_loop079_identity/run484/b_candidate/validation.json'
FINAL='evidence/20260927_loop079_identity/run484/b_candidate/final_admission.json'
REVIEW='evidence/20260927_loop079_identity/run488/astra_frontier_review.md'
HASHES={
 PRIOR:'1fe4480dc6ede2c78d33bcd84986af47311fd19d2dc435b8e57f4a9d7fad8a0a',
 INTERVALS:'bd21d480819f8c66b96691a26ae1bc9cab4d0422c8ac0d98890594604911b708',
 VALID:'67793f2f970da13bf1fb0222d617f951c09941a2efb1ecdd78fbeae9f874a0bb',
 FINAL:'339780afb953ca502ee310c13c3b649c1e30e208fe08a6bfa76f21bbe11e19e4',
 REVIEW:'7d987e2d6c091da0e5663a491b813ede5be49d560d56f7dbdacf2f566347c3d6',
}


def build():
    for name,expected in HASHES.items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('evidence hash mismatch: '+name)
    model=json.loads((ROOT/PRIOR).read_text())
    data=json.loads((ROOT/INTERVALS).read_text())
    valid=json.loads((ROOT/VALID).read_text())
    final=json.loads((ROOT/FINAL).read_text())
    review=(ROOT/REVIEW).read_text()
    if model['model_revision']!='V3.20' or model['current']['accepted_formal_tps']!=571.681:
        raise ValueError('prior Current/model changed')
    if not valid.get('valid') or not final.get('valid') or data['rank_slices']!=40 or data['independent_cohorts']!=5:
        raise ValueError('Run487 admission incomplete')
    if data['capture_origin']!=['startup_full_decode96'] or data['graph_update_branch']!=['after'] or data['hidden_native_leaf_counts']!=[4]:
        raise ValueError('Target graph/hidden branch changed')
    if not all(v is False for v in data['boundary_claims'].values()):
        raise ValueError('reducer boundary promoted')
    if '**PASS for the clean, instrumented, conditional Current-frontier observation.**' not in review:
        raise ValueError('independent review qualification changed')
    if any(model['proof_dag']['certified'].values()):
        raise ValueError('prior proof node unexpectedly certified')

    model['model_revision']='V3.21'
    sched=model['bound_ladder']['scheduling_execution']
    sched['target_full_frontier_local_observation']={
        'status':'admitted_instrumented_conditional_current_only',
        'source':[INTERVALS,VALID,FINAL,REVIEW],
        'scope':'separate exact60 48+12 diagnostic; five correlated all8 cohorts, selected cycle64; startup FULL96 entry bound to Runtime Target; no common all-rank clock',
        'intervals_ms':data['intervals_ms'],
        'cohort_cycles':[data['cohorts'][str(i)]['cycles'] for i in range(1,6)],
        'cohort_useful_tokens':[data['cohorts'][str(i)]['useful_output_tokens'] for i in range(1,6)],
        'graph_update_branch':'after',
        'graph_update_configured_stream_id':102,
        'graph_update_private_stream_relation':'unresolved',
        'hidden_aux_native_allgathers_per_rank':4,
        'R1_captured_producer_completion':'conditional on exact native producer membership and child-stream join',
        'unmarked_current_exposure_ms':None,
        'all_rank_common_clock_makespan_ms':None,
        'necessary_duration_floor_ms':None,
        'removable_e2e_ms':None,
        'matched_A0_B_A1_controls':False,
        'missing':[
            'exact selected graph last-writer/child-stream join for four hidden/aux outputs',
            'actual graph-update backend and effective private-stream producer/consumer join',
            'all-rank arrival and mixed-resource service lower/attainable envelopes',
            'probe overhead and matched unchanged-trajectory A0/B/A1 before original-schedule saving',
        ],
    }
    model['certificate_graph']['scheduling_execution']['target_full_frontier']={
        'status':'instrumented_conditional_current_observation',
        'evidence':[INTERVALS,FINAL,REVIEW],
        'missing':list(sched['target_full_frontier_local_observation']['missing']),
        'bound_effect':'no finite Scheduling latency floor or Product ceiling',
    }
    model['next_measurement']['priority']=(
        'Obtain the exact selected FULL graph entry/generation native debug dump after ordinary drain: bind four hidden/aux output last writers and child-stream joins, plus actual graph-update backend/effective work and consumer join. '
        'Use a targeted native trace only for unresolved producers. In parallel pursue one fresh retained wo_a work witness and authoritative exact-board 910B3 C_plus; neither isolated attained service nor this diagnostic yields a finite ceiling.')
    model['next_measurement']['specific_gate']=(
        'Source-pinned graph dump must match its own acquisition selected entry/generation/model/batch/output addresses without adding hot-path waits; align source/config/shape/branch to Run487, but never equate process-local IDs or device addresses across runs. Retain conditional R1 and private-stream labels until actual producer/join membership is proved. '
        'Then add all-rank critical-path and resource contention evidence before a Scheduling Bound; use formal correctness and repeated E2E for attainable Product claims.')
    model['input_paths'].extend([INTERVALS,VALID,FINAL,REVIEW])
    require_null_endpoints(model)
    if any(model['proof_dag']['certified'].values()):
        raise ValueError('Current observation cannot certify Bound proof node')
    return model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model=build()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='ok',revision='V3.21',finite_endpoints=0)))


if __name__=='__main__':main()
