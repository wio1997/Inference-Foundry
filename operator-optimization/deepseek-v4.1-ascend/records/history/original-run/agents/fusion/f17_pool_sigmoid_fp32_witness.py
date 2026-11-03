#!/usr/bin/env python3
"""CPU arithmetic witness only; no claim about device exp/FMA lowering."""
import json
import numpy as np
s0=np.float32('0.7128095030784607'); s1=np.float32('7.908058166503906')
v0=np.float32('-64.67178344726562'); v1=np.float32('59.7182502746582')
m=np.maximum(s0,s1)
e0=np.exp(np.float32(s0-m)); e1=np.exp(np.float32(s1-m))
d=np.float32(e0+e1)
y0=np.float32(np.float32(e0/d)*v0+np.float32(e1/d)*v1)
p=np.float32(np.float32(1)/(np.float32(1)+np.exp(np.float32(s0-s1))))
y1=np.float32(v0+np.float32(p*np.float32(v1-v0)))
def bf16bits(x):
 u=int(np.asarray(x,dtype=np.float32).view(np.uint32));return (u+0x7fff+((u>>16)&1))>>16
print(json.dumps({'numpy_version':np.__version__,'inputs_fp32':[float(s0),float(s1),float(v0),float(v1)],'max_stabilized_weighted_fp32':float(y0),'sigmoid_centered_fp32':float(y1),'bf16_round_to_nearest_even_hex':[hex(bf16bits(y0)),hex(bf16bits(y1))],'scope':'host NumPy simulated float32 operations; no device executable or original hardware math binding'},indent=2))
assert bf16bits(y0)==0x426f and bf16bits(y1)==0x426e
