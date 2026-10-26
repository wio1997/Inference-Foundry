#!/usr/bin/env python3
"""Advance Resource/Scheduling certificates after Run508/509/511; no numeric promotion."""
import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_22 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT = Path(__file__).resolve().parents[1]
PRIOR = 'evidence/20260927_loop079_identity/run504/bound_calibration_v3_22.json'
VALID = 'evidence/20260927_loop079_identity/run508/b_candidate/validation.json'
UPDATE = 'evidence/20260927_loop079_identity/run508/b_candidate/update_validation.json'
FINAL = 'evidence/20260927_loop079_identity/run508/b_candidate/final_admission.json'
CLEANUP = 'evidence/20260927_loop079_identity/run508/b_candidate/cleanup_status.txt'
RESOURCE_REVIEW = 'evidence/20260927_loop079_identity/run509/astra_resource_certificate_review.md'
SCHED_REVIEW = 'evidence/20260927_loop079_identity/run511/astra_run508_review.md'
SCHED_MANIFEST = 'evidence/20260927_loop079_identity/run511/input_sha256_manifest.json'
HASHES = {
    PRIOR: '994a4aa69be470e4ad3124e7a0ae08e272fa979a88fb4509753c7c345208ed3d',
    VALID: '58418563321a53e82c89092f4d12e20e10a0821ae10caeaf88ab75bb3ad33c20',
    UPDATE: 'f9af293eb2177c02a55ec6659ff358548fdc2e10c4802243d0bdd7a8673263e4',
    FINAL: '26e399a23d5f9460b9fe60880db1bc9c66db0c81c80e939a05542bb777dfff49',
    CLEANUP: 'f8885c889d44d1a1243d45c805ce4932d4835f3967a82e2c5a5cde87589168db',
    RESOURCE_REVIEW: '570616b1fc6405bfa031ef3ae081c3ab98fe74b5d45ad68efd0faf04c8a1812e',
    SCHED_REVIEW: '3fd45eff05eb655c30f8db0d157d05805216f9caaff2b73f5f993dc847df6639',
    SCHED_MANIFEST: 'f803c0507fdb4969b8f2ce7ca579675a0dee9ff26c42801f89d2e21340bb05e0',
}


def build():
    for path, expected in HASHES.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('evidence hash mismatch: ' + path)
    model = build_prior()
    if (json.dumps(model, ensure_ascii=False, indent=2) + '\n') != (ROOT / PRIOR).read_text():
        raise ValueError('V3.22 generator/output mismatch')
    valid = json.loads((ROOT / VALID).read_text())
    update = json.loads((ROOT / UPDATE).read_text())
    final = json.loads((ROOT / FINAL).read_text())
    cleanup = dict(line.split('=', 1) for line in (ROOT / CLEANUP).read_text().splitlines())
    resource_review = (ROOT / RESOURCE_REVIEW).read_text()
    sched_review = (ROOT / SCHED_REVIEW).read_text()
    if model['model_revision'] != 'V3.22' or model['current']['accepted_formal_tps'] != 571.681:
        raise ValueError('prior formal Current/model changed')
    if any(model['proof_dag']['certified'].values()):
        raise ValueError('prior proof DAG unexpectedly certified')
    if not (valid['valid'] and update['valid'] and final['valid'] and valid['run_id'] == update['run_id']):
        raise ValueError('Run508 acquisition admission missing')
    if valid['slices'] != 40 or valid['cohorts'] != 5 or valid['graph_dumps'] != 8 or update['rows'] != 40 or update['ranks'] != 8 or update['cohorts'] != 5:
        raise ValueError('Run508 all-rank scope changed')
    if set(cleanup.values()) != {'0'} or len(cleanup) != 8:
        raise ValueError('Run508 cleanup/final gate incomplete')
    if len(update['summaries']) != 40 or any(
        x['attention_keys'] != 170 or x['counts'] != {'attn_params': 0, 'handles': 0, 'events': 0}
        or x['zip_iterations'] != 0 for x in update['summaries']
    ):
        raise ValueError('selected MLA update-loop observation changed')
    if 'no certified positive W− / matching C+ pair' not in resource_review or 'W−>B' not in resource_review:
        raise ValueError('Resource review exclusion/rate-boundary changed')
    if '**PASS, with the promotion scope below.**' not in sched_review or '**Promote only the selected-call fact:**' not in sched_review:
        raise ValueError('independent Scheduling review scope changed')

    model['model_revision'] = 'V3.23'
    hardware = model['bound_ladder']['hardware_resource']
    hardware['rate_boundary_certificate'] = {
        'status': 'uncertified',
        'source': [RESOURCE_REVIEW],
        'required_form': 'in-window service W <= C_plus*T, or W <= C_plus*T+B with a proved boundary allowance B and W_minus > B',
        'exact_board_C_plus': None,
        'boundary_work_B': None,
        'scope': 'ordinary BF16 dense Cube-only class for the first candidate W_minus; all legal engines must be covered if class broadened',
    }
    model['certificate_graph']['hardware_resource']['rate_boundary_certificate'] = {
        'status': 'uncertified',
        'evidence': [RESOURCE_REVIEW],
        'missing': ['exact-board maximum clock and issue envelope', 'interval cumulative-service guarantee or B', 'matching legal engine class'],
        'bound_effect': 'no positive strict Resource latency floor',
    }
    model['certificate_graph']['hardware_resource']['matching_true_capacity_upper_C_plus']['missing'].append(
        'cumulative service bound for the full formal interval, or proved boundary allowance B with W_minus > B'
    )
    model['certificate_graph']['hardware_resource']['matching_true_capacity_upper_C_plus']['evidence'].append(RESOURCE_REVIEW)
    model['proof_dag']['nodes']['matching_true_C_plus']['missing'].append(
        'matching cumulative interval-service upper, or a proved B boundary term'
    )
    model['proof_dag']['nodes']['matching_true_C_plus']['evidence'].append(RESOURCE_REVIEW)
    model['proof_dag']['nodes']['strict_scope_units_join']['missing'].append(
        'W_minus > B if the only matching capacity statement is W <= C_plus*T+B'
    )
    model['proof_dag']['nodes']['strict_scope_units_join']['evidence'].append(RESOURCE_REVIEW)
    sched = model['bound_ladder']['scheduling_execution']
    sched['run508_selected_MLA_update'] = {
        'status': 'conditional_current_no_loop_work',
        'source': [VALID, UPDATE, FINAL, CLEANUP, SCHED_REVIEW, SCHED_MANIFEST],
        'selected_graph': 'FULL96 Target, five diagnostic cohorts across eight ranks, same-process captured/selected GraphParams and replay entry/generation',
        'actual_update_impl': 'AscendMLAImpl.update_graph_params',
        'attention_key_count_each': 170,
        'attn_params_handles_events_count_each': 0,
        'selected_zip_iterations_each': 0,
        'source_inferred_graph_task_updates_each': 0,
        'source_inferred_event_records_each': 0,
        'stream_context_cost_ms': None,
        'other_private_work': None,
        'four_output_typed_last_writers': None,
        'terminal_notify_to_caller_R1': None,
        'necessary_path_floor_ms': None,
        'removable_product_ms': None,
        'bound_effect': 'remove selected MLA update-loop work from conditional Current DAG for Run508 acquisition only; no finite floor or TPS endpoint',
    }
    model['certificate_graph']['scheduling_execution']['run508_selected_MLA_update'] = {
        'status': 'conditional_no_loop_work',
        'evidence': [VALID, UPDATE, FINAL, CLEANUP, SCHED_REVIEW],
        'missing': ['typed output writers', 'native graph terminal to caller R1', 'all-rank mixed resource service', 'formal E2E transfer'],
        'bound_effect': 'no finite Scheduling latency floor or Product ceiling',
    }
    strict = model['bound_ladder']['product_e2e']['strict_outer_ceiling_sufficient_gate']
    strict['formula'] = ('If interval service W <= C_plus*T, then T >= W_minus/C_plus; '
                         'if only W <= C_plus*T+B, then T >= max(0,W_minus-B)/C_plus and W_minus > B is required for a positive floor. '
                         'TPS <= 49152/T_star only for a positive certified T_star.')
    strict['requirements'] += ' A steady-state rate alone is insufficient without a proved interval boundary term B or cumulative-service guarantee.'
    strict['source'] = [strict['source'], RESOURCE_REVIEW]
    model['certificate_graph']['product_e2e']['strict_outer_ceiling']['scope_checks'].append(
        'C_plus must upper-bound cumulative work throughout the formal interval, or include a proved B and W_minus > B'
    )
    model['next_measurement']['priority'] = (
        'Scheduling: exact installed ReduceMean MIX placeholder ABI and NPUGraph terminal-to-caller semantics, then typed four-output producer/terminal join if source-only proof remains incomplete. '
        'Resource: formal48 first-token request/generation/publication/freshness witness and exact-board C-plus with interval-service/B guarantee. Preserve both tracks before selecting an optimization.'
    )
    model['next_measurement']['specific_gate'] = (
        'Do not reacquire selected MLA update-loop counts unless shape/backend changes. For remaining joins, bind semantic output roles to loaded typed native producers and model completion to caller R1 without extra waits/raw stream getters; carry all-rank resource contention and formal E2E separately.'
    )
    model['input_paths'].extend(HASHES.keys())
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
    print(json.dumps({'status': 'ok', 'revision': 'V3.23', 'finite_endpoints': 0}))


if __name__ == '__main__':
    main()
