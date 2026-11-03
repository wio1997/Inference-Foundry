#!/usr/bin/env python3
"""Exact-integer/Fraction counterexample for the C2 ring read-before-write dependency.

No torch, NPU, target source changes or inference. This models one legal
metadata/score/input assignment to the frozen code, not an observed request.
"""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source_rel = 'sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/compressor/compressor_triton.py'
source = (root / source_rel).read_bytes()
expected = json.loads((root / 'agents/fusion/compressor-engram-input-hashes.json').read_text())[source_rel]
assert sha256(source).hexdigest() == expected, 'frozen compressor source hash differs'

# One request and one valid cache page. Metadata rows are start positions,
# used lengths, reserved output bases, segment input bases, physical pages.
start_pos, used_len, input_base, cache_page = 1, 32, 0, 1
cache_pages, cache_size, width, ratio, max_query_len = 2, 32, 512, 2, 32
assert 0 < cache_page < cache_pages
assert 0 <= max_query_len <= used_len

group_slots = (max_query_len + 1) // 2
local_group = 0
groups_in_batch = (start_pos + used_len) // ratio - start_pos // ratio
assert local_group < groups_in_batch <= group_slots
group_idx = start_pos // ratio + local_group
residual = start_pos - group_idx * ratio
assert residual > 0  # _pool_kernel enters _pooled_blocked(..., HAS_RES=True)

positions = [group_idx * ratio + row for row in range(ratio)]
segment_offsets = [pos - start_pos for pos in positions]
assert positions == [0, 1] and segment_offsets == [-1, 0]
history_pos = positions[0]
history_slot = history_pos % cache_size
first_projection_row = input_base + segment_offsets[1]
output_row = input_base + (group_idx + 1) * ratio - 1 - start_pos
assert first_projection_row == 0 and output_row == 0

write_slots = min(max_query_len, cache_size)
tail_count = min(used_len, cache_size)
write_idx = 31
assert write_idx < min(write_slots, tail_count)
token_in_seg = used_len - tail_count + write_idx
write_projection_row = input_base + token_in_seg
write_pos = start_pos + token_in_seg
write_slot = write_pos % cache_size
assert write_pos == 32 and write_projection_row == 31
assert history_slot == write_slot == 0

# Exact pointer offsets in FP32 elements, for channel 0 of K/V and score.
kv_read = (cache_page * cache_size + history_slot) * 2 * width
score_read = kv_read + width
kv_write = (cache_page * cache_size + write_slot) * 2 * width
score_write = kv_write + width
assert (kv_read, score_read) == (kv_write, score_write) == (32768, 33280)

# Scores are all exactly zero: exp(0)=1, so each probability is exactly 1/2.
# The two vectors are equal in channel 1, and differ in channel 0 only.
old_history = [Fraction(2), Fraction(3)]
first_current = [Fraction(4), Fraction(3)]
last_current = [Fraction(10), Fraction(3)]
def pool(a, b):
    return [(u + v) / 2 for u, v in zip(a, b)]
correct = pool(old_history, first_current)
wrong_if_early_overwrite = pool(last_current, first_current)
assert correct == [3, 3] and wrong_if_early_overwrite == [7, 3]
# Remaining 510 channels are selected zero. The source-level witness shows
# different pre-norm BF16 pooled values (all these integers are representable).

def exact(values): return [str(x) for x in values]
result = {
    'kind': 'hypothetical_source_semantics_witness_not_observed_169',
    'source_path': source_rel, 'source_sha256': expected,
    'metadata_example': {'start_pos': start_pos, 'used_len': used_len, 'input_base': input_base,
                         'page': cache_page, 'num_cache_pages': cache_pages, 'max_query_len': max_query_len},
    'parameters': {'ratio': ratio, 'cache_rows_per_page': cache_size, 'width': width},
    'pool': {'group_slots': group_slots, 'groups_in_request': groups_in_batch, 'group_index': group_idx,
             'positions': positions, 'segment_offsets': segment_offsets, 'history_ring_slot': history_slot,
             'output_row': output_row},
    'cache_update': {'tail_count': tail_count, 'write_idx': write_idx, 'input_projection_row': write_projection_row,
                     'position': write_pos, 'ring_slot': write_slot},
    'addresses_fp32_elements': {'pool_history_kv': kv_read, 'pool_history_score': score_read,
                                'update_kv': kv_write, 'update_score': score_write},
    'channel_0_and_1_bf16_pooled': {'original_read_before_write': exact(correct),
                                    'hypothetical_write_before_read': exact(wrong_if_early_overwrite)},
    'scope_limits': ['only shows a pre-norm semantic difference under stated hypothetical inputs',
                     'does not prove actual request metadata, addresses, implementation timings, or end-to-end effect',
                     'does not prove every pair of pool/cache operations must serialize'],
    'no_device_work': True,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
