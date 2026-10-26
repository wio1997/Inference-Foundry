#!/usr/bin/env python3
"""Run621 host-only validator fixtures for descriptor admission."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from loop081_cache_descriptor_validate_run621 import validate_records

OUT = Path('/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run621/preflight/validator_selftest.json')
RUN_TS = 'LOOP081-RUN621-FAKE'


def desc(seed: int) -> dict[str, object]:
    return {'status': 'tensor', 'device': 'npu:0', 'format_id': 2,
            'dtype': 'torch.bfloat16', 'shape': [8, 16, 1, 192],
            'stride': [3072, 192, 192, 1], 'storage_nbytes': 49152,
            'storage_cdata': seed, 'storage_base_ptr': 100000 + seed,
            'tensor_data_ptr': 100000 + seed, 'storage_offset_elements': 0}


def record(cycle: int) -> dict[str, object]:
    return {'status': 'descriptors_only_no_device_values', 'cycle': cycle,
            'rank': 0, 'cohort': 5, 'run_ts': RUN_TS,
            'layers': [{'layer': str(layer), 'cache_layer_prefix': f'x.{layer}',
                        'block_size': 16, 'cache_owner_object_id': layer,
                        'cache': desc(layer)} for layer in (43, 44, 45)],
            'common': {name: desc(100 + idx) for idx, name in enumerate((
                'query_start_loc', 'seq_lens', 'slot_mapping', 'block_table_tensor'))}}


positive = [record(64), record(65)]
assert validate_records(positive, 0, RUN_TS)['cycles'] == [64, 65]
rejected = []


def reject(name: str, mutate) -> None:
    sample = copy.deepcopy(positive)
    mutate(sample)
    try:
        validate_records(sample, 0, RUN_TS)
    except (ValueError, KeyError, TypeError):
        rejected.append(name)
    else:
        raise AssertionError(name + ' wrongly admitted')


reject('missing_second_cycle', lambda x: x.pop())
reject('repeated_cycle', lambda x: x[1].update(cycle=64))
reject('wrong_rank', lambda x: x[0].update(rank=7))
reject('wrong_cohort', lambda x: x[0].update(cohort=6))
reject('wrong_run_ts', lambda x: x[1].update(run_ts='STALE'))
reject('missing_layer', lambda x: x[0]['layers'].pop())
reject('storage_generation_changed',
       lambda x: x[1]['layers'][0]['cache'].update(storage_cdata=999))
reject('view_pointer_changed',
       lambda x: x[1]['layers'][0]['cache'].update(tensor_data_ptr=999))
reject('view_offset_changed',
       lambda x: x[1]['layers'][0]['cache'].update(storage_offset_elements=4))
reject('cache_owner_changed',
       lambda x: x[1]['layers'][0].update(cache_owner_object_id=999))
reject('missing_common', lambda x: x[0]['common'].pop('seq_lens'))
reject('non_npu_cache',
       lambda x: x[0]['layers'][0]['cache'].update(device='cpu'))
reject('unknown_cache_format',
       lambda x: x[0]['layers'][0]['cache'].update(format_id=None))
reject('nontensor_cache',
       lambda x: x[0]['layers'][0]['cache'].update(status='not_tensor'))
OUT.parent.mkdir(parents=True, exist_ok=True)
result = {
    'status': 'descriptor_validator_positive_and_14_negative_fixtures_pass',
    'validator_sha256': hashlib.sha256(Path(
        '/data/wio/Inference_Foundry/scripts/loop081_cache_descriptor_validate_run621.py'
    ).read_bytes()).hexdigest(),
    'rejected': rejected,
    'scope': 'Host-only synthetic schema tests; no actual W0/NPU/service or Bound',
}
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'negative_count': len(rejected)}))
