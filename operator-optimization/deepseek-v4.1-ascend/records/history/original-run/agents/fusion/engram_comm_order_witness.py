#!/usr/bin/env python3
"""Conditional ring all-gather vs all-reduce payload arithmetic; no NPU."""
from fractions import Fraction

p, heads, dim, outputs, tokens = 8, 24, 256, 25600, 3
assert heads % p == 0 and outputs % p == 0
local_heads = heads // p
local_features = local_heads * dim
gather_elements = (p-1) * local_features
reduce_elements = 2 * (p-1) * (outputs // p)
ratio=Fraction(reduce_elements,gather_elements)
assert (local_features, gather_elements,reduce_elements,ratio)==(768,5376,44800,Fraction(25,3))
print('per rank, one direction, per token:',gather_elements,'gather elements;',reduce_elements,'all-reduce elements; ratio',ratio)
print('three tokens, BF16 bytes per rank, one direction:',tokens*gather_elements*2,tokens*reduce_elements*2)
