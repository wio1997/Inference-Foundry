#!/usr/bin/env python3
"""V3.43: source-gated next acquisition for fixed-work Resource/Scheduling Bound."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
E = ROOT / 'evidence/20260928_loop081_bound'
INPUTS = {
    'v3_42': (E / 'run622/bound_calibration_v3_42.json',
              '41b52c3dd7b63924e930b3f0e56bb0f9f4289cb10a8ef63105b5d547a6041c02'),
    'scheduling_review': (E / 'run623/astra_next_packet_review.md',
                          'f42965ff744c86587f984f7ed85a7341e99519e16617aac87b6d68d1b72c35dc'),
    'resource_review': (E / 'run623/astra_resource_next_review.md',
                        '03f750ad136b4048dbb638a4b1dc1cb614149709ac9948078174391f1787b1e3'),
    'installed_method_manifest': (E / 'run623/resource_method_sources/manifest.json',
                                  '9841bc9264f75e2c7d54a6d5f27d3380eec4dc7457a8063d40b5c932ef7ff8ee'),
}


def need(condition: bool, why: str) -> None:
    if not condition:
        raise ValueError(why)


for key, (path, expected) in INPUTS.items():
    need(hashlib.sha256(path.read_bytes()).hexdigest() == expected, key + ' SHA drift')

model = json.loads(INPUTS['v3_42'][0].read_text())
prior = model['bound_semantics_v3_42']
need(prior['formal_current_tps'] == 571.681, 'formal Current drift')
for key in ('strict_resource_hardware_floor_s',
            'strict_scheduling_execution_floor_s',
            'strict_product_e2e_tps_ceiling',
            'numeric_current_to_credible_limit_gap'):
    need(prior[key] is None, 'prior strict endpoint drift: ' + key)

model['model_revision'] = 'v3.43_fixed_work_actual_invocation_gate_no_capacity_certificate'
model['bound_semantics_v3_43'] = {
    'active_objective': 'same DSpark7 acceptance, cycle count, output semantics and model work; minimize full Runtime execution time',
    'formal_current_tps': 571.681,
    'resource_hardware': {
        'compulsory_work_complete': False,
        'compulsory_HBM_bytes': None,
        'exact_board_certified_cumulative_C_plus_B': None,
        'strict_floor_s': None,
        'installed_environment': 'CANN9.1.0, driver file26.0.rc1; these are package identities, not booted image attestation',
        'current_or_attained_only': 'Run247-249 counters, Run577-580 fixture service, HCCL Test algorithm bandwidth and PMU profiling cannot certify compulsory bytes or universal service upper capacity',
        'next_numerator_probe': 'One actually consumed, unoverwritten context projection with input/model/quant, semantic prefix, cache generation, actual reader inclusion, downstream relevance and complete entry-result credit; conditional partial W_minus only',
        'capacity_access': 'Run462 OEM exact-bin maximum envelope access and Run590 image linkage remain unresolved; avoid repeating peak tests as C_plus certificates',
    },
    'scheduling_execution': {
        'strict_floor_s': None,
        'actual_invocation_hook_gate': 'Context formatted scatter; query prefill/decode and alternate multistream scatter; prefill/decode final comp_ratio<=1 attention kwargs, branch-specific metadata and exact native flow in same acquisition',
        'row_generation_witness': None,
        'current_storage_hazard_if_seen': 'current storage ordering only; alternative versioning/renaming may remove it unless value dependency and resource cost show necessity',
        'observer_payload': 'Run622 12,288 B/rank is six candidate rows at post-context plus pre-reader only; index/metadata and possible pre-context old row are separate',
        'live_status': 'Run623 source recommendation NOT LIVE-READY; preflight bounded metadata, validity, lifetime, original-stream ordering, negative fixtures and all8 correctness first',
    },
    'conditional_engineering': {
        'status': 'scenario/sensitivity only, not a strict hardware ceiling',
        'next_modeling_step': 'resource-constrained full Product dependency DAG sensitivity with current intervals and separately measured attainable mixed-service cells; choose future interference measurements by largest output sensitivity, not kernel hotspot',
        'numeric_interval_tps': None,
    },
    'product_e2e': {
        'strict_ceiling_tps': None,
        'numeric_current_to_credible_limit_gap': None,
        'reason': 'Incomplete necessary work/traffic, certified exact-board C_plus/B, legal full critical path and formal same-W0 transfer',
    },
    'strict_resource_hardware_floor_s': None,
    'strict_scheduling_execution_floor_s': None,
    'strict_product_e2e_tps_ceiling': None,
    'numeric_current_to_credible_limit_gap': None,
    'evidence_class': 'Run623 independent source/methodology design review; no new service/NPU or formal E2E experiment',
}
v = model['bound_semantics_v3_43']
need(all(v[k] is None for k in ('strict_resource_hardware_floor_s',
                               'strict_scheduling_execution_floor_s',
                               'strict_product_e2e_tps_ceiling',
                               'numeric_current_to_credible_limit_gap')) and
     all(v['resource_hardware'][k] is None for k in
         ('compulsory_HBM_bytes', 'exact_board_certified_cumulative_C_plus_B',
          'strict_floor_s')) and
     all(v['scheduling_execution'][k] is None for k in
         ('strict_floor_s', 'row_generation_witness')) and
     v['conditional_engineering']['numeric_interval_tps'] is None and
     all(v['product_e2e'][k] is None for k in
         ('strict_ceiling_tps', 'numeric_current_to_credible_limit_gap')),
     'unsupported Bound promotion rejected')

out = E / 'run623/bound_calibration_v3_43.json'
out.write_text(json.dumps(model, indent=2) + '\n')
print(json.dumps({'status': model['model_revision'], 'strict_endpoints': 'null'}))
