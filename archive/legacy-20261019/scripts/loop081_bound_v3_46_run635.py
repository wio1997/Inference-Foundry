#!/usr/bin/env python3
"""Run635: scope-controlled preparation Host coverage in fixed-work Bound V3.46."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
INPUTS = {
    'v3_45': ('evidence/20260928_loop081_bound/run632/bound_calibration_v3_45.json', 'e823cf22e249238d33c04a6ea4d1e8661ed4dfefb57b26f794db607da0502ec0'),
    'prep': ('evidence/20260928_loop081_bound/run633/prep_host_coverage.json', '1bac88d687cbd16b9a92123d07f7af2140c845ab356c0eb8954183e1ea0feea0'),
    'prep_review': ('evidence/20260928_loop081_bound/run633/astra_prep_coverage_review.md', '53e04749492b89f3b63d06ae8c4e7bcf6f29856f0c0dbf49f9b1c5d06cfa0c76'),
    'mode': ('evidence/20260928_loop081_bound/run634/prep_mode_coverage.json', 'c173d9fc92756e9185157828d815948a4338cd8467d5a67b587e2efe78b402bc'),
    'mode_review': ('evidence/20260928_loop081_bound/run634/astra_prep_mode_review.md', '3f4c3b26c98ed3ebbc05ad6907d8e4802ca81b0eb9342285e0618a38670035a2'),
}
OUT = ROOT / 'evidence/20260928_loop081_bound/run635/bound_calibration_v3_46.json'


def read(name):
    path, expected = INPUTS[name]
    payload = (ROOT / path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise AssertionError(f'source drift: {name}')
    return json.loads(payload) if path.endswith('.json') else None


def validate_no_bound(model):
    path = ROOT / 'scripts/loop081_bound_v3_45_run632.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != '7da466646e310f31d02a52a59c17c9709b7cea219e69524a4bf91f23a09ad0bf':
        raise AssertionError('V3.45 validator source drift')
    spec = importlib.util.spec_from_file_location('v345', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_no_bound(model)
    if 'bound_semantics_v3_46' in model:
        v = model['bound_semantics_v3_46']
        for section, names in {
            'current_diagnostic_preparation': ('formal_transfer',),
            'resource_hardware': ('necessary_target_NONE_work_and_traffic',
                                  'attainable_mixed_compute_HBM_HCCL', 'strict_floor_s'),
            'scheduling_execution': ('target_NONE_device_input_output_readiness',
                                     'all8_existing_wait_and_first_consumer',
                                     'exposed_critical_path_s', 'strict_floor_s'),
            'product_e2e': ('strict_ceiling_tps',
                            'conditional_engineering_interval_tps',
                            'numeric_current_to_credible_limit_gap'),
        }.items():
            if any(v[section][name] is not None for name in names):
                raise AssertionError(f'unproved V3.46 evidence or endpoint: {section}')


def main():
    model, prep, mode = read('v3_45'), read('prep'), read('mode')
    read('prep_review')
    read('mode_review')
    validate_no_bound(model)
    if prep['status'] != 'run606_same_W0_preparation_host_call_coverage_only' or mode['status'] != 'run606_same_W0_current_Target_Host_dispatch_mode_join_only':
        raise AssertionError('diagnostic source status')
    if mode['source_sha256']['run633_prep'] != INPUTS['prep'][1] or len(prep['rows']) != 32 or len(mode['rows']) != 32:
        raise AssertionError('prep/mode linkage')
    if any(r['own_NONE_count'] + r['own_FULL_count'] + r['prior_FULL_target_count'] not in (9, 12, 8, 11) for r in mode['rows']):
        raise AssertionError('mode count')
    by_rank = [[r for r in mode['rows'] if r['rank'] == rank] for rank in range(8)]
    if any({r['cohort'] for r in rows} != {5, 6, 7, 8} or
           sum(r['own_NONE_count'] for r in rows) != 24 or
           sum(r['own_FULL_count'] for r in rows) != 13 or
           sum(r['prior_FULL_target_count'] for r in rows) != 3
           for rows in by_rank):
        raise AssertionError('fixed same-W0 dispatch census')
    model['model_revision'] = 'v3.46_fixed_work_prep_host_coverage_no_strict_bound'
    model['bound_semantics_v3_46'] = {
        'active_objective': 'same DSpark7 acceptance/cycles/output/model work; minimize full Product Runtime execution time',
        'source_pins': {name: {'path': p, 'sha256': h} for name, (p, h) in INPUTS.items()},
        'current_diagnostic_preparation': {
            'run': 606, 'same_W0_within_run': True, 'observer_perturbed': True,
            'all8_rank_cohort_rows': 32,
            'target_host_dispatch_by_cohort': mode['summary'],
            'per_rank_own_target_NONE_count': 24,
            'per_rank_own_target_FULL_count': 13,
            'per_rank_prior_target_FULL_count': 3,
            'host_coverage_scope': 'Run633 exact request-ID ownership and interval unions; Target/propose Host intervals do not overlap within one rank. These are Python call intervals, not device execution, HCCL or critical-path savings.',
            'mode_scope': 'Run634 raw frontier validates NONE/FULL Target dispatch. FULL Host return can include existing current-stream synchronize. NONE duration can include necessary prefill, mixed device work and waits.',
            'formal_transfer': None,
        },
        'resource_hardware': {'necessary_target_NONE_work_and_traffic': None,
                              'attainable_mixed_compute_HBM_HCCL': None,
                              'strict_floor_s': None},
        'scheduling_execution': {'target_NONE_device_input_output_readiness': None,
                                 'all8_existing_wait_and_first_consumer': None,
                                 'exposed_critical_path_s': None,
                                 'strict_floor_s': None},
        'product_e2e': {'strict_ceiling_tps': None,
                        'conditional_engineering_interval_tps': None,
                        'numeric_current_to_credible_limit_gap': None,
                        'next_action': 'Preflight same-W0 low-overhead full-Product causal boundaries: actual ordinary Target NONE input ready, logits and aux producer completion, related HCCL side streams, existing waits, first actual consumer and output publication; OFF/ON/OFF timing transfer. Typed KV remains supporting.'},
        'scope_rule': 'No subtraction of observer-perturbed Host intervals from formal Run99 wall, no claim that Target NONE is dominant removable time, and no algorithmic acceptance/cycle change.',
    }
    validate_no_bound(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'revision': model['model_revision'], 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
