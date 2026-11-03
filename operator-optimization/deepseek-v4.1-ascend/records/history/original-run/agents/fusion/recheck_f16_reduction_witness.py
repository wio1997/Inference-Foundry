#!/usr/bin/env python3
"""CPU-only IEEE binary32 rounding witness; not a device kernel test."""
import json
import struct

def round_f32(x):
    return struct.unpack('>f', struct.pack('>f', x))[0]

# The BF16-representable products for one output coordinate. All other
# entries among the 4096 elements can be zero under the hypothetical input.
a, b, c = map(round_f32, (2**40, -(2**40), 1))
left = round_f32(round_f32(a + b) + c)
right = round_f32(a + round_f32(b + c))
assert left == 1.0 and right == 0.0
# Both resulting values are exactly BF16 representable.
print(json.dumps({'left_tree_fp32_then_bf16': left,
                  'right_tree_fp32_then_bf16': right,
                  'hypothetical_only': True}, sort_keys=True))
