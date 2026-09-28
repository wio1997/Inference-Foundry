#!/usr/bin/env python3
"""Run621 all8 actual-W0 cache descriptor admission, no device values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def validate_records(records: object, rank: int, run_ts: str) -> dict[str, object]:
    need(isinstance(records, list) and len(records) == 2, 'exact two cycle records')
    caches = {}
    metadata = {}
    for expected_cycle, record in zip((64, 65), records):
        need(record['status'] == 'descriptors_only_no_device_values' and
             record['cycle'] == expected_cycle and record['rank'] == rank and
             record['cohort'] == 5 and record['run_ts'] == run_ts,
             'identity/cycle/role drift')
        layers = record['layers']
        need(len(layers) == 3 and [r['layer'] for r in layers] == ['43', '44', '45'],
             'active layer geometry drift')
        for layer in layers:
            desc = layer['cache']
            need(desc['status'] == 'tensor' and desc['device'].startswith('npu:') and
                 isinstance(desc['format_id'], int) and
                 all(isinstance(v, int) and v > 0 for v in desc['shape']) and
                 desc['storage_nbytes'] > 0 and desc['storage_cdata'] > 0 and
                 desc['storage_base_ptr'] > 0 and desc['tensor_data_ptr'] > 0 and
                 desc['storage_offset_elements'] >= 0 and layer['block_size'] > 0,
                 'cache descriptor missing/invalid')
            key = (expected_cycle, layer['layer'])
            caches[key] = {
                'owner_object_id': layer['cache_owner_object_id'],
                'storage_cdata': desc['storage_cdata'],
                'base_ptr': desc['storage_base_ptr'],
                'tensor_data_ptr': desc['tensor_data_ptr'],
                'storage_offset_elements': desc['storage_offset_elements'],
                'format_id': desc['format_id'],
                'dtype': desc['dtype'],
                'shape': desc['shape'],
                'stride': desc['stride'],
                'storage_nbytes': desc['storage_nbytes'],
                'block_size': layer['block_size'],
                'prefix': layer['cache_layer_prefix'],
            }
        for name in ('query_start_loc', 'seq_lens', 'slot_mapping',
                     'block_table_tensor'):
            desc = record['common'][name]
            need(desc['status'] == 'tensor' and desc['storage_nbytes'] > 0,
                 f'common {name} descriptor missing')
            metadata[(expected_cycle, name)] = {
                'device': desc['device'], 'dtype': desc['dtype'],
                'shape': desc['shape'], 'storage_nbytes': desc['storage_nbytes'],
                'format_id': desc['format_id'],
            }
    for layer in ('43', '44', '45'):
        a, b = caches[(64, layer)], caches[(65, layer)]
        need(all(a[field] == b[field] for field in (
            'owner_object_id', 'storage_cdata', 'base_ptr', 'tensor_data_ptr',
            'storage_offset_elements', 'format_id',
            'dtype', 'shape', 'stride', 'storage_nbytes', 'block_size', 'prefix')),
             f'cache owner/storage replaced across cycles layer{layer}')
    return {
        'rank': rank,
        'cycles': [64, 65],
        'cache': {layer: caches[(64, layer)] for layer in ('43', '44', '45')},
        'common_metadata': {name: metadata[(64, name)] for name in (
            'query_start_loc', 'seq_lens', 'slot_mapping', 'block_table_tensor')},
        'scope': 'Actual cache/metadata tensor descriptors and two observed matching cache view descriptors; no lifetime certificate, values or row generations',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm-dir', required=True, type=Path)
    parser.add_argument('--run-ts', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for rank in range(8):
        path = args.arm_dir / 'product' / f'rank{rank}_cohort5.json'
        need(path.is_file(), f'missing rank{rank} product')
        payload = json.loads(path.read_text())
        need(payload['rank'] == rank and payload['cohort'] == 5 and
             payload['run_ts'] == args.run_ts,
             f'rank{rank} Product identity')
        rows.append(validate_records(payload.get('run621_cache_descriptors'),
                                     rank, args.run_ts))
    result = {
        'status': 'all8_actual_W0_two_cycle_cache_descriptors_only_pass',
        'run_ts': args.run_ts,
        'rank_count': len(rows),
        'rows': rows,
        'scope': ('Run621 inherited guarded 48+48 fixed-work diagnostic. Actual cache '
                  'owner/storage geometry, format and metadata descriptors only. No '
                  'device slot/index values, loaded attention binary, native task join, '
                  'row writer/read generation, fresh W-minus, strict Bound, formal E2E.'),
        'strict_resource_floor_s': None,
        'strict_scheduling_floor_s': None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'rank_count': len(rows)}))


if __name__ == '__main__':
    main()
