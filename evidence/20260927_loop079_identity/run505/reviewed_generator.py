#!/usr/bin/env python3
"""Calibrate the two Bound tracks from Run497/502/503, without numeric promotion."""
import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT = Path(__file__).resolve().parents[1]
PRIOR = 'evidence/20260927_loop079_identity/run489/bound_calibration_v3_21.json'
FIRST = 'evidence/20260927_loop079_identity/run497/first_position_cpu.json'
VALID = 'evidence/20260927_loop079_identity/run502/b_candidate/validation.json'
FINAL = 'evidence/20260927_loop079_identity/run502/b_candidate/final_admission.json'
DUMP = 'evidence/20260927_loop079_identity/run502/b_candidate/graph_dump_admission.json'
META = 'evidence/20260927_loop079_identity/run502/b_candidate/graph_dump/rank0_cohort5_acl_graph.meta.json'
REVIEW = 'evidence/20260927_loop079_identity/run503/astra_run502_graph_review.md'
MANIFEST = 'evidence/20260927_loop079_identity/run503/input_hashes.json'
HASHES = {
    PRIOR: '269bf2182ef61bce200470b671a50e4124689156a1350c384ecf1f0be0612bf8',
    FIRST: '2c78373b7dbba68ad839cf9fe8c2575db14fb50ccd676b98d36fb616dd10e21c',
    VALID: '0fc2de5755a597322121f555fc65185f8b1d3dd46161eff960b24632480d060a',
    FINAL: '339780afb953ca502ee310c13c3b649c1e30e208fe08a6bfa76f21bbe11e19e4',
    DUMP: '4fd3f6ed2dfaf4561a8e5765ea85b8cf266224467309c946e97602fbe4305615',
    META: 'a0083baa46b93affe7d7886ca9a4edb35e952281b19c97e5c0113aff9c7c0d98',
    REVIEW: '92aeeb078c0511e3d3969be52be8f4965e87f65bde7dbcfcb920f2e21db95a86',
    MANIFEST: '272bd6ebbe1b71158f7e064652379ddc90f8802086f1a1d999f1767aca3a9c12',
}


def build():
    for path, expected in HASHES.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('evidence hash mismatch: ' + path)
    model = json.loads((ROOT / PRIOR).read_text())
    first = json.loads((ROOT / FIRST).read_text())
    valid = json.loads((ROOT / VALID).read_text())
    final = json.loads((ROOT / FINAL).read_text())
    dump = json.loads((ROOT / DUMP).read_text())
    meta = json.loads((ROOT / META).read_text())
    review = (ROOT / REVIEW).read_text()
    if model['model_revision'] != 'V3.21' or model['current']['accepted_formal_tps'] != 571.681:
        raise ValueError('prior formal Current/model changed')
    if any(model['proof_dag']['certified'].values()):
        raise ValueError('prior proof DAG unexpectedly certified')
    if not (valid['valid'] and final['valid'] and dump['ranks'] == 8 and dump['meta_join']):
        raise ValueError('Run502 acquisition admission missing')
    if valid['slices'] != 40 or valid['cohorts'] != 5 or valid['graph_dumps'] != 8:
        raise ValueError('Run502 scope changed')
    if meta['node_count'] != 5412 or meta['stream_task_counts'] != {'0': 1040, '1': 3985, '99': 387}:
        raise ValueError('graph schema changed')
    backend = meta['graph_update_backend']
    selected = backend['selected_actual_update']
    if backend['impl']['source']['qualname'] != 'AscendMLAImpl' or selected['branch'] != 'after' or selected['configured_update_stream']['stream_id'] != 102 or selected['return_count'] != 1:
        raise ValueError('selected update backend changed')
    if '**ACCEPT Run502 as a valid, source-bound, post-drain graph diagnostic and conditional Current structure evidence.**' not in review:
        raise ValueError('independent qualification changed')
    if '**Do not yet admit four proven last writers, effective private-stream no-work/completion, or an unconditional producer→caller-R1 certificate.**' not in review:
        raise ValueError('independent exclusion changed')
    # CPU lemma is admitted only as a proof-scope reduction, never as in-window W-minus.
    if first.get('source_identity') != 'pass' or first.get('equality_patterns') != 128:
        raise ValueError('first-position CPU lemma did not pass')

    model['model_revision'] = 'V3.22'
    algorithm = model['bound_ladder']['algorithm_resource']
    algorithm['first_position_work_witness'] = {
        'status': 'conditional_scope_reduction_only',
        'source': [FIRST, REVIEW],
        'identity': 'for an active slot, sampled first token equals same-cycle first Target argmax for all seven-draft match patterns',
        'conditional_dense_wo_a_operations': 8388608,
        'in_window_external_retention_certified': False,
        'fresh_dense_execution_certified': False,
        'compulsory_work_subset_operations': None,
        'missing': ['same-request/generation external token lineage', 'formal-window freshness', 'actual loaded final-layer ordinary dense work and no legal reuse/recompute alternative'],
    }
    hardware = model['bound_ladder']['hardware_resource']
    hardware['run502_graph_task_census'] = {
        'status': 'current_exporter_structure_only',
        'source': [DUMP, META, REVIEW],
        'tasks_per_rank': 5412,
        'streams': {'0': 1040, '1': 3985, '99': 387},
        'memcpy_async_tasks_per_rank': 43,
        'memcpy_bytes': None,
        'compulsory_traffic_bytes': None,
        'exact_board_aggregate_capacity_upper': None,
        'qualification': 'No task count, Args text or configured clock is a compulsory-traffic or matching maximum-capacity certificate.',
    }
    sched = model['bound_ladder']['scheduling_execution']
    sched['run502_graph_dependency'] = {
        'status': 'conditional_current_structure_only',
        'source': [VALID, FINAL, DUMP, META, REVIEW, MANIFEST],
        'selected_graph': 'same-process FULL96 Target startup entry/generation/model/batch/output owner joined to Runtime; dump after ordinary count drain',
        'actual_update_impl': 'AscendMLAImpl.update_graph_params',
        'configured_update_stream_id': 102,
        'actual_update_iteration_count': None,
        'external_event_handle_generation_join': None,
        'four_output_typed_last_writers': None,
        'main_candidate': 'RmsNorm stream1 task2944, ABI-supported candidate, not a last-writer certificate',
        'aux_candidates': 'ReduceMean stream1 tasks2788/2848/2925, address maxima with unresolved mixed-core placeholder ABI',
        'candidate_event_dag': 'all5412 tasks can reach terminal notify under per-stream Task Id and event-suffix assumptions; exporter provides no measured duration',
        'terminal_notify_to_caller_R1': None,
        'run487_instrumented_replay_envelope_transfer': 'not transferred to Run502; retained in its prior standalone Current observation',
        'necessary_path_floor_ms': None,
        'removable_product_ms': None,
        'missing': ['typed four-output loaded writers and overlapping writes', 'same-generation external event/handle/native task correlation', 'model terminal notify to caller-R1 continuation', 'all-rank arrival plus mixed resource contention and attainable service', 'formal E2E trajectory transfer and marker overhead controls'],
    }
    model['certificate_graph']['scheduling_execution']['run502_graph_dependency'] = {
        'status': 'conditional_structure_only',
        'evidence': [VALID, FINAL, DUMP, META, REVIEW],
        'missing': list(sched['run502_graph_dependency']['missing']),
        'bound_effect': 'no finite Scheduling latency floor or Product ceiling',
    }
    model['next_measurement']['priority'] = (
        'Source-only decode the exact installed ReduceMean MIX placeholder ABI and NPUGraph model terminal-to-caller semantics. '
        'Then one minimal dynamic acquisition only for unresolved same-generation MLA update iteration/event/handle and typed four-output producer joins. '
        'In parallel preserve the Resource W-minus/external first-token and authoritative exact-board C-plus tracks; do not substitute attained service or graph task counts.'
    )
    model['next_measurement']['specific_gate'] = (
        'Use original frozen 48+12 shape/cycle selector and source hashes; record actual update key/zip/API counts and capture-to-replay event/handle generation, plus semantic producer descriptors and loaded native task correlation. '
        'No extra wait or raw stream getter. Require all-rank dependency/resource contention and formal correctness/repeated E2E before any attainable Product endpoint.'
    )
    model['input_paths'].extend([FIRST, VALID, FINAL, DUMP, META, REVIEW, MANIFEST])
    require_null_endpoints(model)
    if any(model['proof_dag']['certified'].values()):
        raise ValueError('diagnostic evidence cannot certify a Bound proof node')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'revision': 'V3.22', 'finite_endpoints': 0}))


if __name__ == '__main__':
    main()
