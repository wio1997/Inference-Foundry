#!/usr/bin/env python3
"""Run622: reduce admitted Run621 actual-cache geometry to observer budgets.

Allocated tensor storage and hypothetical copied operands are not compulsory
HBM traffic. No new service or NPU work.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
INPUT = ROOT / 'evidence/20260928_loop081_bound/run621/live/b/descriptor_admission.json'
OUTPUT = ROOT / 'evidence/20260928_loop081_bound/run622/cache_geometry_budget.json'
PIN = '0e78e7b9258de04aa40f6b2ef95e89fb0dd9f6281b431161a613e34be8d43424'


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


need(hashlib.sha256(INPUT.read_bytes()).hexdigest() == PIN, 'Run621 admission SHA drift')
source = json.loads(INPUT.read_text())
need(source['status'] == 'all8_actual_W0_two_cycle_cache_descriptors_only_pass' and
     source['rank_count'] == 8 and len(source['rows']) == 8,
     'Run621 admission drift')

rows = []
for rank, record in enumerate(source['rows']):
    need(record['rank'] == rank and record['cycles'] == [64, 65], 'rank/cycle drift')
    cache = record['cache']
    need(set(cache) == {'43', '44', '45'}, 'layer coverage drift')
    allocated = 0
    storage_tokens = []
    for layer in ('43', '44', '45'):
        item = cache[layer]
        need(item['format_id'] == 2 and item['dtype'] == 'torch.bfloat16' and
             item['shape'] == [34090, 32, 1, 512] and
             item['stride'] == [16384, 512, 512, 1] and
             item['block_size'] == 32,
             'cache geometry/format drift')
        row_bytes = item['shape'][2] * item['shape'][3] * 2
        logical_allocation = math.prod(item['shape']) * 2
        need(row_bytes == 1024 and logical_allocation == item['storage_nbytes'],
             'cache byte geometry drift')
        need(item['base_ptr'] == item['tensor_data_ptr'] and
             item['storage_offset_elements'] == 0,
             'cache view is not base')
        storage_tokens.append((item['storage_cdata'], item['base_ptr']))
        allocated += item['storage_nbytes']
    need(len(set(storage_tokens)) == 3, 'intra-rank cache allocation alias')
    meta = record['common_metadata']
    need(meta['slot_mapping']['shape'] == [96] and
         meta['block_table_tensor']['shape'] == [12, 32768] and
         meta['slot_mapping']['dtype'] == 'torch.int32' and
         meta['block_table_tensor']['dtype'] == 'torch.int32',
         'metadata geometry drift')
    rows.append({
        'rank': rank,
        'cache_layers': 3,
        'retained_cache_storage_bytes_observed': allocated,
        'one_cache_row_logical_bytes': 1024,
        'two_cycles_three_layers_one_row_each_copy_bytes': 2 * 3 * 1024,
        'two_cycles_three_layers_pre_post_copy_bytes': 2 * 3 * 2 * 1024,
        'slot_map_view_logical_bytes': 96 * 4,
        'slot_map_backing_storage_bytes': meta['slot_mapping']['storage_nbytes'],
        'full_block_table_view_logical_bytes': 12 * 32768 * 4,
        'block_table_backing_storage_bytes': meta['block_table_tensor']['storage_nbytes'],
        'cache_format_id': 2,
    })

need({x['retained_cache_storage_bytes_observed'] for x in rows} == {3 * 1117061120},
     'all8 storage-size parity')
need({x['slot_map_backing_storage_bytes'] for x in rows} == {33152} and
     {x['block_table_backing_storage_bytes'] for x in rows} == {2097152},
     'all8 metadata backing-size parity')
result = {
    'status': 'all8_actual_W0_cache_geometry_budget_observed',
    'source_sha256': PIN,
    'contract': 'fixed DSpark7 8x910B3 DP1TP8 48x32K->1024 c12',
    'rank_count': 8,
    'rows': rows,
    'all8_retained_three_layer_cache_storage_bytes_observed': sum(
        r['retained_cache_storage_bytes_observed'] for r in rows),
    'one_selected_row_each_layer_two_cycles_under_128KiB_per_rank': all(
        r['two_cycles_three_layers_pre_post_copy_bytes'] <= 128 * 1024 for r in rows),
    'full_block_table_snapshot_rejected_as_unselective': True,
    'scope': ('Actual W0 cache descriptor and selected-row scratch arithmetic only. '
              'The 12KiB/rank pre/post example is a hypothetical copy budget, not '
              'evidence of selected valid rows or a live buffer allocation. Full '
              'block-table is 1.573MB logical view/rank; capture branch-specific '
              'selected entries instead. Backing storage is not compulsory HBM '
              'traffic, physical allocation, or Scheduling/Resource latency.'),
    'actual_slot_values': None,
    'actual_reader_eligible_rows': None,
    'compulsory_HBM_bytes': None,
    'strict_resource_floor_s': None,
    'strict_scheduling_floor_s': None,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'rank_count': len(rows),
                  'per_rank_cache_bytes': rows[0]['retained_cache_storage_bytes_observed'],
                  'example_pre_post_row_scratch_bytes': rows[0]['two_cycles_three_layers_pre_post_copy_bytes']}))
