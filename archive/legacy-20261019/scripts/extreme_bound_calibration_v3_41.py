#!/usr/bin/env python3
"""V3.41 fixed-work Bound: scoped current intervals and conditional SWA reader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
E = ROOT / 'evidence/20260928_loop081_bound'
INPUTS = {
    'base_v3_40': (E / 'run613/bound_calibration_v3_40.json',
                   'b48c02d069cd31209b94f2bc08b9f53f3e41be3281d80ba0542b153f187c5bfd'),
    'packet_gate': (E / 'run614/typed_kv_packet_gate.json',
                    '73ed975c9bab9eb47ea877741f223e62e9eda1de94c56d76010ef259c3d85f79'),
    'resource_review': (E / 'run614/astra_resource_numeric_review.md',
                        'aad8e6f2eb5e76b6e0ae8d6cfda82e6c19c219ff1eb217bd20f514b4c9bd682c'),
    'snapshot_smoke': (E / 'run615/snapshot_api_smoke.json',
                       '902439f0fe5fe46079504b5d9e53b3ff49ef26d7f6df4e6a7a3247490aa198d2'),
    'observer_preflight': (E / 'run615/astra_observer_preflight.md',
                           '7e5d16abe4974d64024acfeac3815bce9c2bf91458993989e917c9b062ec2517'),
    'intervals': (E / 'run616/eager_interval_census.json',
                  'a6030a53c42f7683f1323c61494387aa525b1dc40cdc7a830b94318e191c323e'),
    'reader_review': (E / 'run616/astra_reader_abi_review.md',
                      '10ec319edb9ee02368c6b341fdf761328e561c31d9003485a82479ba08bda2b8'),
    'oracle': (E / 'run619/swa_eligible_domain.json',
               '51869101193a2efa2240f83f2abb19d0063e19dd5dd00d6f5e4527d0eb6d37a0'),
    'oracle_final_review': (E / 'run619/astra_oracle_final_review.md',
                            '3be6d5a3c0be0ef8c61657d335965d63d4b8c83d566aca9d3f57c9f60ec8be58'),
}


def need(value: bool, why: str) -> None:
    if not value:
        raise ValueError(why)


for label, (path, expected) in INPUTS.items():
    need(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
         f'{label} SHA drift')

model = json.loads(INPUTS['base_v3_40'][0].read_text())
intervals = json.loads(INPUTS['intervals'][0].read_text())
oracle = json.loads(INPUTS['oracle'][0].read_text())
need(model['bound_semantics_v3_40']['formal_current_tps'] == 571.681,
     'formal current drift')
need(intervals['record_count'] == 72 and
     intervals['status'] == 'all8_exact_flow_current_native_start_intervals_only',
     'current interval admission drift')
need(oracle['fixture_count'] == 22 and oracle['actual_W0_eligible_rows'] is None,
     'conditional reader admission drift')

model['model_revision'] = 'v3.41_fixed_work_conditional_reader_not_live_bound'
model['bound_semantics_v3_41'] = {
    'active_objective': 'minimum Runtime execution time for unchanged DSpark7 acceptance, cycle trajectory, output semantics and model work',
    'formal_current_tps': 571.681,
    'current_instrumented_exact_flow_spacing': {
        'same_W0_72_source_order_candidate_triplets': 72,
        'context_to_query_native_start_us': intervals['distributions_us']['context_to_query_native_start_us'],
        'query_to_reader_native_start_us': intervals['distributions_us']['query_to_reader_native_start_us'],
        'scope': 'Run610/611 profiler-perturbed native starts, not data dependency, removable time, unmarked service, or formal Run99 transfer',
    },
    'conditional_swa_eligible_reader': {
        'source_pinned_CPU_fixtures': 22,
        'dense_branch': 'inclusive window mapped by actual request block table',
        'sparse_branch': 'first-negative prefix of physical slot IDs; block table required by tiling but its values are not used in sparse address mapping',
        'actual_W0_values': None,
        'metadata_execution_coverage': None,
        'loaded_op_binary_and_tiling_identity': None,
        'physical_HBM_reads': None,
        'scope': 'Run619 pure CPU helper after Astra Run617/618 challenges; not a live-ready or executed-reader witness',
    },
    'observer_preflight': 'Run615 descriptor and same-stream toy snapshot PASS; Astra implementation design PASS but NOT LIVE-READY. Need bounded actual current-W0 typed allocation/content/row generations, metadata and ABI args, exact native flow, predecessor/successor, correctness and cleanup.',
    'resource_review': 'Astra High Run614 numeric audit rejects promoting 21.344GB current counter/attained fixture service to compulsory traffic or exact-board C_plus,B; strict Resource remains null.',
    'strict_resource_hardware_floor_s': None,
    'strict_scheduling_execution_floor_s': None,
    'strict_product_e2e_tps_ceiling': None,
    'numeric_current_to_credible_limit_gap': None,
    'highest_information_next_probe': 'Guarded bounded actual-W0 typed KV invocation packet with current cache storage lifetime, row writer epochs, slot/index/block-table/metadata snapshots and loaded-op identity; admit Scheduling identity separately from conditional fresh W_minus. Independently seek exact-board certified upper cumulative C_plus,B.',
    'independent_review': 'Run614 Resource and Run615/616/619 Astra High scope checks; no finite endpoint or E2E intervention',
}
strict = ('strict_resource_hardware_floor_s', 'strict_scheduling_execution_floor_s',
          'strict_product_e2e_tps_ceiling', 'numeric_current_to_credible_limit_gap')
need(all(model['bound_semantics_v3_41'][field] is None for field in strict),
     'strict endpoint promotion rejected')
out = E / 'run619/bound_calibration_v3_41.json'
out.write_text(json.dumps(model, indent=2) + '\n')
print(json.dumps({'status': model['model_revision'], 'strict_endpoints': 'null',
                  'triplets': 72, 'fixtures': 22}))
