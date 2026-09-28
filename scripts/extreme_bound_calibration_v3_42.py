#!/usr/bin/env python3
"""V3.42 fixed-work Bound: observed SWA cache geometry, not compulsory traffic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
E = ROOT / 'evidence/20260928_loop081_bound'
INPUTS = {
    'base_v3_41': (E / 'run619/bound_calibration_v3_41.json',
                   '7c0e70267fafc8bb08ba63f2dd35d78e3afffad9e9742b62943aa68303434162'),
    'run621_summary': (E / 'run621/admission_summary.json',
                       '19af86275bae790455ebd50c0f831211c06910316e0a51091af6c289694879af'),
    'run621_descriptors': (E / 'run621/live/b/descriptor_admission.json',
                           '0e78e7b9258de04aa40f6b2ef95e89fb0dd9f6281b431161a613e34be8d43424'),
    'run621_astra_post': (E / 'run621/astra_post_review.md',
                          'b0f131f873086df4c2f6c221d9c89796d358bb46cd56fab75be6ae1cbeb822da'),
    'run622_budget': (E / 'run622/cache_geometry_budget.json',
                      '8b0519762f0f43fd36a704c8bbca4038f242a059376e9d54eeab86234a353e09'),
}


def need(value: bool, why: str) -> None:
    if not value:
        raise ValueError(why)


for label, (path, expected) in INPUTS.items():
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    need(got == expected, f'{label} SHA drift: {got}')

model = json.loads(INPUTS['base_v3_41'][0].read_text())
summary = json.loads(INPUTS['run621_summary'][0].read_text())
descriptor = json.loads(INPUTS['run621_descriptors'][0].read_text())
budget = json.loads(INPUTS['run622_budget'][0].read_text())
need(model['bound_semantics_v3_41']['formal_current_tps'] == 571.681,
     'formal Current drift')
need(summary['status'] == 'run621_all8_guarded_cache_descriptor_only_admitted' and
     summary['admissions']['descriptor']['sha256'] == INPUTS['run621_descriptors'][1] and
     set(summary['cleanup'].values()) == {'0'} and
     summary['server_post_count_before_after_stop'] == [96, 96],
     'Run621 admission/cleanup drift')
need(descriptor['rank_count'] == 8 and len(descriptor['rows']) == 8 and
     budget['rank_count'] == 8 and len(budget['rows']) == 8 and
     budget['source_sha256'] == INPUTS['run621_descriptors'][1],
     'all8 descriptor/budget drift')
need(all(row['cycles'] == [64, 65] and set(row['cache']) == {'43', '44', '45'}
         for row in descriptor['rows']), 'cache sample drift')
need(all(row['retained_cache_storage_bytes_observed'] == 3351183360 and
         row['one_cache_row_logical_bytes'] == 1024 and
         row['two_cycles_three_layers_pre_post_copy_bytes'] == 12288
         for row in budget['rows']), 'budget drift')
need(budget['actual_slot_values'] is None and
     budget['actual_reader_eligible_rows'] is None and
     budget['compulsory_HBM_bytes'] is None and
     budget['strict_resource_floor_s'] is None and
     budget['strict_scheduling_floor_s'] is None,
     'conditional evidence promoted')

model['model_revision'] = 'v3.42_fixed_work_observed_cache_geometry_not_traffic'
model['bound_semantics_v3_42'] = {
    'active_objective': 'minimum Runtime execution time for unchanged DSpark7 acceptance, cycle trajectory, output semantics and model work',
    'formal_current_tps': 571.681,
    'actual_W0_cache_descriptor_observation': {
        'rank_count': 8,
        'sampled_cycles': [64, 65],
        'swa_layers': [43, 44, 45],
        'cache_format_id': 2,
        'cache_dtype': 'torch.bfloat16',
        'cache_shape': [34090, 32, 1, 512],
        'cache_storage_bytes_each_layer_rank': 1117061120,
        'three_layer_storage_bytes_per_rank': 3351183360,
        'one_logical_row_bytes': 1024,
        'block_table_view_logical_bytes_per_rank': 1572864,
        'block_table_backing_storage_bytes_per_rank': 2097152,
        'slot_mapping_view_logical_bytes_per_rank': 384,
        'slot_mapping_backing_storage_bytes_per_rank': 33152,
        'scope': 'Run621 observed matching two-cycle descriptor/view identities, not continuous allocation lifetime; common metadata captured before refresh_common, not actual attention invocation values. Storage capacity/backing bytes are not compulsory HBM traffic.',
    },
    'observer_payload_budget': {
        'hypothetical_two_cycles_three_layers_one_row_pre_post_bytes_per_rank': 12288,
        'selected_operand_cap_bytes_per_rank': 131072,
        'actual_selected_valid_rows': None,
        'full_observer_cost_bytes': None,
        'scope': 'Run622 arithmetic only; no live row selection, scratch allocation, metadata generation, or latency witness. Select branch-specific block-table entries instead of full-table copy.',
    },
    'resource_hardware_bound': {
        'compulsory_cache_HBM_bytes': None,
        'certified_exact_board_C_plus_B': None,
        'floor_s': None,
        'reason': 'Run621 storage capacity and Run247 actual counter traffic do not establish compulsory work/traffic or certified cumulative all8 capability.',
    },
    'scheduling_execution_bound': {
        'typed_writer_reader_generation': None,
        'actual_attention_reader_eligible_rows': None,
        'producer_consumer_critical_path': None,
        'floor_s': None,
        'reason': 'Two matching descriptors do not identify slot values, writer/read generations, native invocation metadata or legal overlap.',
    },
    'product_e2e_bound': {
        'ceiling_tps': None,
        'numeric_current_to_credible_limit_gap': None,
        'reason': 'No finite Resource/Scheduling floor or formal same-W0 E2E intervention; Run621 is diagnostic.',
    },
    'strict_resource_hardware_floor_s': None,
    'strict_scheduling_execution_floor_s': None,
    'strict_product_e2e_tps_ceiling': None,
    'numeric_current_to_credible_limit_gap': None,
    'highest_information_next_probe': 'Guarded actual invocation-level branch/slot/index and selected block-table entry snapshot, one bounded valid KV row pre/post with original-stream generation and exact native attention flow; separately obtain exact-board certified C_plus,B. Require negative fixtures and independent preflight.',
    'independent_review': 'Run621 Astra High scoped live preflight/post review and Run622 arithmetic audit; V3.42 awaits independent review.',
}
strict = ('strict_resource_hardware_floor_s', 'strict_scheduling_execution_floor_s',
          'strict_product_e2e_tps_ceiling', 'numeric_current_to_credible_limit_gap')
v = model['bound_semantics_v3_42']
need(all(v[field] is None for field in strict) and
     all(v['resource_hardware_bound'][field] is None for field in
         ('compulsory_cache_HBM_bytes', 'certified_exact_board_C_plus_B', 'floor_s')) and
     all(v['scheduling_execution_bound'][field] is None for field in
         ('typed_writer_reader_generation', 'actual_attention_reader_eligible_rows',
          'producer_consumer_critical_path', 'floor_s')) and
     all(v['product_e2e_bound'][field] is None for field in
         ('ceiling_tps', 'numeric_current_to_credible_limit_gap')),
     'strict or nested endpoint promotion rejected')
out = E / 'run622/bound_calibration_v3_42.json'
out.write_text(json.dumps(model, indent=2) + '\n')
print(json.dumps({'status': model['model_revision'], 'strict_endpoints': 'null',
                  'rank_count': 8, 'sampled_cycles': [64, 65]}))
