#!/usr/bin/env python3
"""Exact proof that positive float32 sqrt can merge two unequal float32 values.

Hypothetical finite IEEE round-to-nearest-even model. This is not a sample from 169
or a proof of the implementation's softplus, top-k, or tie semantics.
"""
from fractions import Fraction as F
import json,struct
bits=[0x47800000,0x47800001] # 65536 and next binary32 value
x=[F(struct.unpack('>f',struct.pack('>I',v))[0]) for v in bits]
s0=F(256);su=F(struct.unpack('>f',struct.pack('>I',0x43800001))[0]);mid=(s0+su)/2
assert x==[F(65536),F(65536)+F(1,128)] and x[0]<x[1]<mid*mid
assert x[0]==s0*s0 and (mid*mid-x[1])==F(1,4294967296)
print(json.dumps({'input_float32_bits':[f'{v:08x}' for v in bits],
 'exact_inputs':[str(a) for a in x],'lower_root':'256', 'next_float32_root':str(su),
 'upper_input_below_squared_midpoint_by':str(mid*mid-x[1]),
 'conclusion':'under correct IEEE FP32 sqrt round-to-nearest-even both results are 256.0 despite unequal inputs',
 'limitations':['illustrates non-injectivity of rounded sqrt only, not target sigmoid/softplus kernel or actual logits',
                'tie-breaking and top-k dependence must be established for the actual operator','no NPU work']},indent=2))
