#!/usr/bin/env python3
"""CPU-only numeric witnesses for a restricted frozen F09 semantics; no NPU."""
from fractions import Fraction
from pathlib import Path
import hashlib, json, struct

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'agents/fusion/frozen-source/vllm_ascend/ops/triton/quantize_indexer_query.py'
expected = '6065b79f0f764e99bafae71347c8c8dfd0625ad52b0d695639c47e71338d0bba'
actual = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
if actual != expected: raise RuntimeError('frozen source checksum changed')
f32 = lambda x: struct.unpack('>f', struct.pack('>f', x))[0]
f16 = lambda x: struct.unpack('>e', struct.pack('>e', x))[0]

def reference(v):
    a = max(map(abs, v))
    scale = max(f16(f32(f32(a)/f32(127.0))), 2**-24)
    q = [max(-127, min(127, round(f32(f32(x)/f32(scale))))) for x in v]
    return scale, q

zero = [0.] * 128
s0, q0 = reference(zero)
if s0 != 2**-24 or any(q0): raise RuntimeError('zero witness mismatch')
expected_scale = Fraction(129,8192)
for i in range(128):
    v = zero.copy(); v[i] = 2.0
    s, q = reference(v)
    if Fraction(s) != expected_scale or q[i] != 127 or any(q[j] for j in range(128) if j != i):
        raise RuntimeError(f'coordinate witness mismatch: {i}')
abs_div = f32(2/127)
if Fraction(abs_div) != Fraction(2113665,134217728): raise RuntimeError('unexpected binary32 division witness')
result = {'scope':'128 BF16 input coordinates each from {0,2}; zero vector and 128 single-coordinate variants',
          'source_sha256':actual,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'zero_scale_f16_bits':struct.pack('>e',s0).hex(), 'zero_scale_exact':str(Fraction(s0)),
          'single_two_abs_div_f32_bits':struct.pack('>f',abs_div).hex(),
          'single_two_scale_f16_bits':struct.pack('>e',float(expected_scale)).hex(),
          'single_two_scale_exact':str(expected_scale),'single_two_quantized_value':127,
          'distinguishing_coordinates':128,'bf16_zero_bits':'0000','bf16_two_bits':'4000',
          'reference_precision':'binary32 round-to-nearest, binary16 conversion round-to-nearest-ties-to-even, input 0/2 exact; CPU math not actual NPU execution',
          'limitations':['Proof is a conditional exposed-output interface and restricted input/access domain, not model-output behavior.',
                         'No statement of measured HBM bytes, actual runtime dtype or loaded compiler rounding behavior.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
