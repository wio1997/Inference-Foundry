"""CPU IEEE-style model, not a torch/CANN or NPU execution test."""
import numpy as np

bits = np.arange(65536, dtype=np.uint32)
x = (bits << np.uint32(16)).view(np.float32)
finite = (bits & np.uint32(0x7F80)) != np.uint32(0x7F80)
scale_python = 128**-0.5 * 32**-0.5
scale_fp32 = np.float32(scale_python)
assert scale_fp32 == np.float32(1 / 64)
with np.errstate(over='ignore', under='ignore', invalid='ignore'):
    scaled = x * scale_fp32
    original = scaled.astype(np.float16).view(np.uint16)
    # IEEE nearest-even BF16 rounding of the FP32 product. The round-up
    # formula applies to finite values, including signed zeros/subnormals.
    u = scaled.view(np.uint32)
    bf16_bits_in_fp32 = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    alternative = bf16_bits_in_fp32.view(np.float32).astype(np.float16).view(np.uint16)
    bad = np.flatnonzero(finite & (original != alternative))
# Check a half-way-to-minimum FP16 subnormal and its next BF16 input,
# including sign of zero. These are independent expected-bit checks.
for input_bf16, expected_fp16 in [(0x3600, 0x0000), (0xB600, 0x8000), (0x3601, 0x0001), (0xB601, 0x8001)]:
    assert int(original[input_bf16]) == expected_fp16
    assert int(alternative[input_bf16]) == expected_fp16
print('FP16 underflow ties and signed-zero checks: passed')
print('python scale:', repr(scale_python))
print('float32 scale:', repr(scale_fp32))
print('finite BF16 codes:', int(finite.sum()))
print('finite bitwise mismatches:', len(bad))
assert len(bad) == 0
