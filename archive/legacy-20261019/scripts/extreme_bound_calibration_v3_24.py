#!/usr/bin/env python3
"""Add exact-version general Graph completion evidence without native identity promotion."""
import argparse
import hashlib
import json
from pathlib import Path

from extreme_bound_calibration_v3_23 import build as build_prior
from extreme_bound_calibration_v3_16_rev2 import require_null_endpoints

ROOT = Path(__file__).resolve().parents[1]
PRIOR = 'evidence/20260927_loop079_identity/run512/bound_calibration_v3_23.json'
REVIEW = 'evidence/20260927_loop079_identity/run510/astra_terminal_abi_source_review.md'
SOURCE_MANIFEST = 'evidence/20260927_loop079_identity/run510/runtime_source_manifest.json'
ARGUMENTS = 'evidence/20260927_loop079_identity/run510/raw_argument_checks.json'
INSTALLED = 'evidence/20260927_loop079_identity/run510/installed_files.json'
HASHES = {
    PRIOR: 'f7e9fc401b5e678305e321a4fbefc0d64640c9ae4c60eabba43a7d96ae736c88',
    REVIEW: '120565abe8270dbb218554c5482548208808dfabe5e6b5a4f6630124dc916685',
    SOURCE_MANIFEST: '493e03d02ad59737eb760b64282004e98c49b40e813167b51f15b9d525d1b7ac',
    ARGUMENTS: '41aaccb990d302ac677cdf50a3e4b450d87169413ec2b4d3bc856e5bf2028a07',
    INSTALLED: 'd82aa7ddb14b52ac8af4d0ec1907dd17654b86654073de65a20d4133fba2bbac',
}


def build():
    for path, expected in HASHES.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('evidence hash mismatch: ' + path)
    model = build_prior()
    if json.dumps(model, ensure_ascii=False, indent=2) + '\n' != (ROOT / PRIOR).read_text():
        raise ValueError('V3.23 generator/output mismatch')
    review = (ROOT / REVIEW).read_text()
    if '**PASS for the source findings below; NOT a complete Run502 typed-last-writer or Scheduling certificate.**' not in review:
        raise ValueError('Run510 source review admission changed')
    if '**Keep conditional for Run502:**' not in review or 'not a certified decoder' not in review:
        raise ValueError('Run510 native identity/ABI exclusion changed')
    if model['model_revision'] != 'V3.23' or model['current']['accepted_formal_tps'] != 571.681 or any(model['proof_dag']['certified'].values()):
        raise ValueError('prior formal Current/proof state changed')

    model['model_revision'] = 'V3.24'
    sched = model['bound_ladder']['scheduling_execution']
    sched['run510_terminal_and_MIX_source'] = {
        'status': 'version_matched_general_contract_only',
        'source': [REVIEW, SOURCE_MANIFEST, ARGUMENTS, INSTALLED],
        'cann_public_release': 'v9.1.0 tag 0b4af8a31c75a1f1f43638a8f75b9e3c6182e031',
        'installed_build': 'CANN 9.1.0 V100R001C11SPC001B243',
        'release_mechanism': 'model endGraphNotify is recorded on model stream; execution stream waits after model submit in reviewed Stars/non-AICPU branches',
        'exporter_timing': 'synthetic per-task constants; not measured duration',
        'placeholder_label_consumes_argument_bytes': False,
        'run502_reduce_mean_aux_candidate_raw_word': 3,
        'run502_reduce_mean_aux_candidate_raw_byte_offset': 24,
        'run502_reduce_mean_aux_typed_y_role': None,
        'run508_loaded_runtime_branch_and_build_join': None,
        'run508_terminal_notify_object_join': None,
        'run508_four_output_last_writers': None,
        'run508_caller_R1_completion_join': None,
        'necessary_path_floor_ms': None,
        'bound_effect': 'source-supported general completion mechanism narrows the identity search; no native/run-specific completion or numeric Bound',
    }
    model['certificate_graph']['scheduling_execution']['run510_terminal_and_MIX_source'] = {
        'status': 'general_source_contract_only',
        'evidence': [REVIEW, SOURCE_MANIFEST, ARGUMENTS, INSTALLED],
        'missing': ['exact loaded B243 backend/build branch', 'same-generation end-notify object and caller R1 order', 'typed loaded MIX output ABI and exhaustive four-output overlapping writes'],
        'bound_effect': 'no finite Scheduling latency floor or Product ceiling',
    }
    model['proof_dag']['nodes']['typed_necessary_path']['evidence'].append(REVIEW)
    model['proof_dag']['nodes']['typed_necessary_path']['missing'].append(
        'bind release general end-notify mechanism to selected loaded backend, exact model/notify generation and typed four-output producers'
    )
    model['next_measurement']['priority'] = (
        'Scheduling source/identity: exact installed B243 runtime branch/endGraphNotify join, exact compiled ReduceMean MIX launch layout and typed four-output native producer/overlap inventory. '
        'Then all-rank resource-constrained dependency and formal Product transfer. Resource formal48 fresh retained W-minus and matching interval-service C-plus/B remain independent.'
    )
    model['next_measurement']['specific_gate'] = (
        'Use authoritative build/ABI record or narrow native correlation for the exact selected model/notify/execution stream; semantic aux mean and final norm descriptors must bind to loaded tasks and all overlapping writers. '
        'No broad timing rerun or synthetic exporter duration. Keep Current571.681 and all finite endpoints null until their own proof gates.'
    )
    model['input_paths'].extend(HASHES.keys())
    require_null_endpoints(model)
    if any(model['proof_dag']['certified'].values()):
        raise ValueError('source review cannot certify a Bound proof node')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    model = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'ok', 'revision': 'V3.24', 'finite_endpoints': 0}))


if __name__ == '__main__':
    main()
