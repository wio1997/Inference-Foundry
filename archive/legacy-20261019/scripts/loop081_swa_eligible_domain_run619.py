#!/usr/bin/env python3
"""Run619: bounded source-pinned SWA row mapper and CPU fixtures.

This computes a logical eligible cache-slot set for *supplied* input values.
It is not an actual Run610 input capture, physical HBM-load census, or Bound.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend')
OUT = Path('/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run619/swa_eligible_domain.json')
SOURCES = {
    'swa_kernel': (ROOT / 'csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_kernel.h',
                   'fc6cff0a2e18ba4c124aeac7e2e9ab2746cc0a5954848e7e556778a5e898c7a0'),
    'copy_helper': (ROOT / 'csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_common.h',
                    '044b59865853d8c5a69083e81a39939be527486d016fa57dcf4b3e47c828c8b4'),
    'swa_cube': (ROOT / 'csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_cube.h',
                 '40c541ed870552bac0fdea0578da975ba07560f4ece44596c767192ce5eced48'),
    'tiling': (ROOT / 'csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp',
               '98fb6d201fd7657fe44c4ebe9a2445af377cb4b5db47861f0b52db2a8b8dacf1'),
    'tiling_header': (ROOT / 'csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.h',
                      '0a89fb2ab87434a914f1e0286f455ae0f18555f2848101a253f487fae0c774c1'),
    'callsite': (ROOT / 'vllm_ascend/attention/dsa_v1.py',
                 '88a0cd69dbbba476d0ac179431568590e6e876134390fba77f22721069f491cb'),
}


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def bounded_int(value: object, name: str, limit: int = (1 << 31) - 1) -> int:
    need(type(value) is int and 0 <= value <= limit, name + ' must be bounded integer')
    return value


for name, (path, expected) in SOURCES.items():
    need(hashlib.sha256(path.read_bytes()).hexdigest() == expected, name + ' SHA drift')
kernel = SOURCES['swa_kernel'][0].read_text()
helper = SOURCES['copy_helper'][0].read_text()
cube = SOURCES['swa_cube'][0].read_text()
tiling_header = SOURCES['tiling_header'][0].read_text()
need('if (constInfo.hasOriSparseIndices)' in kernel and
     'GetOriSparseActualSeqLen()' in kernel and
     'tempLoopInfo.oriMaskLeft = 0;' in kernel and
     'tempLoopInfo.actOriS2Size - tempLoopInfo.actS1Size' in kernel and
     'DataCopyPABySlots<KV_T>' in cube and
     'DataCopyPABySlots(LocalTensor<T>' in helper and
     'slotId) / shape.blockSize' in helper and
     'SPARSE_LIMIT = 2048' in tiling_header,
     'source semantics anchor drift')


def eligible_slots(*, kv_len: int, q_len: int, q_index: int, block_size: int,
                   block_table: list[int] | None, win_left: int, win_right: int,
                   sparse_indices: list[int] | None = None,
                   cache_slot_capacity: int | None = None) -> list[int]:
    """SWA PA_ND one-query ABI eligibility, excluding alignment/padding loads.

    Sparse indices are physical slot IDs; only entries before the first
    negative sentinel in the min(width, kv_len) prefix are semantically valid.
    The continuous-window branch resolves logical positions through the actual
    request's block table. This function never substitutes a guessed table.
    """
    for name, value in (('kv_len', kv_len), ('q_len', q_len), ('q_index', q_index),
                        ('block_size', block_size), ('win_left', win_left),
                        ('win_right', win_right)):
        bounded_int(value, name)
    if cache_slot_capacity is not None:
        bounded_int(cache_slot_capacity, 'cache_slot_capacity')
    need(kv_len >= q_len > 0 and 0 <= q_index < q_len and block_size > 0,
         'invalid sequence or block geometry')
    need(win_left >= 0 and win_right >= 0, 'negative window')
    # The current tiling requires this optional argument even in the sparse
    # branch, whose *values* are not used by DataCopyPABySlots.
    need(block_table is not None, 'tiling requires ori_block_table')
    need(isinstance(block_table, list) and len(block_table) > 0 and
         all(type(x) is int and -(1 << 31) <= x <= (1 << 31) - 1
             for x in block_table), 'block table int32 schema')
    if sparse_indices is not None:
        need(isinstance(sparse_indices, list) and
             all(type(slot) is int and -(1 << 31) <= slot <= (1 << 31) - 1
                 for slot in sparse_indices), 'sparse int32 schema')
        need(128 <= len(sparse_indices) <= 2048 and len(sparse_indices) % 128 == 0,
             'unsupported sparse index width')
        output = []
        for slot in sparse_indices[:min(len(sparse_indices), kv_len)]:
            if slot < 0:
                break
            if cache_slot_capacity is not None:
                need(slot < cache_slot_capacity, 'sparse physical slot outside cache')
            output.append(slot)
        return output
    query_logical_position = kv_len - q_len + q_index
    need(query_logical_position + win_right <= (1 << 31) - 1,
         'window right expression exceeds signed int32')
    left = max(0, query_logical_position - win_left)
    right = min(kv_len - 1, query_logical_position + win_right)
    output = []
    for position in range(left, right + 1):
        logical_block = position // block_size
        need(logical_block < len(block_table), 'stale/short block table')
        physical_block = block_table[logical_block]
        need(physical_block >= 0, 'invalid physical block')
        slot = physical_block * block_size + position % block_size
        if cache_slot_capacity is not None:
            need(slot < cache_slot_capacity, 'dense physical slot outside cache')
        output.append(slot)
    return output


def storage_relative_bytes(slot: int, *, block_size: int, kv_stride0_elements: int,
                           head_dim: int, element_size: int, view_offset_elements: int,
                           storage_nbytes: int, actual_npu_format: str) -> tuple[int, int]:
    """Source PA_ND, one KV head, ND view address interval for one full row."""
    need(actual_npu_format == 'ND', 'unsupported physical NPU format')
    for name, value in (('slot', slot), ('block_size', block_size),
                        ('kv_stride0_elements', kv_stride0_elements),
                        ('head_dim', head_dim), ('element_size', element_size),
                        ('view_offset_elements', view_offset_elements),
                        ('storage_nbytes', storage_nbytes)):
        bounded_int(value, name, (1 << 63) - 1)
    need(slot >= 0 and block_size > 0 and head_dim > 0 and element_size > 0 and
         view_offset_elements >= 0 and kv_stride0_elements >= block_size * head_dim,
         'invalid layout geometry')
    first_element = (view_offset_elements + slot // block_size * kv_stride0_elements +
                     slot % block_size * head_dim)
    first_byte = first_element * element_size
    end_byte = first_byte + head_dim * element_size
    need(end_byte <= storage_nbytes and end_byte <= (1 << 63) - 1,
         'row beyond bounded storage')
    return first_byte, end_byte


fixtures = []


def check(name: str, expected: list[int], **kwargs) -> None:
    observed = eligible_slots(**kwargs)
    need(observed == expected, f'{name}: {observed} != {expected}')
    fixtures.append({'name': name, 'status': 'pass', 'eligible_slots': observed})


base = dict(kv_len=10, q_len=2, q_index=0, block_size=4,
            block_table=[7, 3, 9], win_left=3, win_right=0,
            cache_slot_capacity=40)
check('dense_window_paged', [13, 14, 15, 36], **base)
check('dense_second_query', [14, 15, 36, 37], **{**base, 'q_index': 1})
check('sparse_physical_slots_ignore_window_and_table', [9, 3],
      **{**base, 'sparse_indices': [9, 3, -1, 8] + [-1] * 124, 'block_table': [0, 0, 0]})
check('sparse_prefix_capped_by_kv_len', [9, 3],
      **{**base, 'kv_len': 2, 'sparse_indices': [9, 3, 8] + [-1] * 125, 'block_table': [0, 0, 0]})
check('sparse_first_sentinel_empty', [],
      **{**base, 'sparse_indices': [-1, 9] + [-1] * 126, 'block_table': [0, 0, 0]})
check('sparse_duplicate_preserves_multiplicity', [9, 9],
      **{**base, 'sparse_indices': [9, 9, -1] + [-1] * 125, 'block_table': [0, 0, 0]})
need(storage_relative_bytes(13, block_size=4, kv_stride0_elements=100,
                            head_dim=2, element_size=2, view_offset_elements=8,
                            storage_nbytes=4096, actual_npu_format='ND') == (620, 624),
     'view-offset/stride address fixture')
fixtures.append({'name': 'noncontiguous_stride_and_view_offset', 'status': 'pass',
                 'storage_relative_byte_range': [620, 624]})


def reject(name: str, **kwargs) -> None:
    try:
        eligible_slots(**kwargs)
    except ValueError:
        fixtures.append({'name': name, 'status': 'rejected'})
        return
    raise AssertionError(name + ' failed to reject')


reject('short_block_table', **{**base, 'block_table': [7]})
reject('invalid_physical_block', **{**base, 'block_table': [7, -1, 9]})
reject('missing_dense_table', **{**base, 'block_table': None})
reject('missing_sparse_table_even_if_unused',
       **{**base, 'sparse_indices': [9] + [-1] * 127, 'block_table': None})
reject('sparse_outside_cache', **{**base, 'sparse_indices': [41] + [-1] * 127})
reject('invalid_query_row', **{**base, 'q_index': 2})
reject('K_shorter_than_Q_unsupported', **{**base, 'kv_len': 1})
reject('sparse_bad_width', **{**base, 'sparse_indices': [9, -1]})
reject('sparse_above_tiling_limit', **{**base, 'sparse_indices': [9] + [-1] * 2175})
reject('block_table_int32_overflow', **{**base, 'block_table': [1 << 40, 3, 9]})
reject('window_right_int32_overflow', **{**base, 'win_right': (1 << 31) - 1})
reject('float_sparse_slot_rejected', **{**base, 'sparse_indices': [9.5] + [-1] * 127})
reject('bool_sparse_slot_rejected', **{**base, 'sparse_indices': [True] + [-1] * 127})
reject('overflow_sparse_slot_rejected', **{**base, 'sparse_indices': [1 << 31] + [-1] * 127})
try:
    storage_relative_bytes(13, block_size=4, kv_stride0_elements=100, head_dim=2,
                           element_size=2, view_offset_elements=8, storage_nbytes=4096,
                           actual_npu_format='NZ')
except ValueError:
    fixtures.append({'name': 'unknown_npu_format_rejected', 'status': 'rejected'})
else:
    raise AssertionError('unknown NPU format failed to reject')

result = {
    'status': 'source_pinned_conditional_abi_domain_fixture_pass',
    'source_sha256': {name: expected for name, (_, expected) in SOURCES.items()},
    'scope': ('Pure CPU row-mapping helper with selected tiling guards, not a complete ABI '
              'validator. Toy block_size4/head_dim2 address test does not represent supported '
              'production geometry. PA_ND SWA semantic eligible cache slots require actual '
              'callsite values are supplied. Sparse branch uses physical slot IDs and '
              'sentinel prefix, dense branch paged continuous window. No actual W0 values, '
              'loaded binary/tiling identity, physical read timestamp/bytes, generation '
              'edge, or numerical Bound.'),
    'fixtures': fixtures,
    'fixture_count': len(fixtures),
    'actual_W0_eligible_rows': None,
    'physical_HBM_bytes': None,
    'strict_resource_floor_s': None,
    'strict_scheduling_floor_s': None,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'fixtures': len(fixtures)}))
