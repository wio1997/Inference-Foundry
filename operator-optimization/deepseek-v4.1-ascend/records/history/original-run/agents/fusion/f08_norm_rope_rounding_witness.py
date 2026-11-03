#!/usr/bin/env python3
"""Exact arithmetic counterexample, CPU only; not an emulation of an NPU kernel."""
from fractions import Fraction as F

def bf16_normal(x):
    if not x:
        return x
    sign = -1 if x < 0 else 1
    a = abs(x)
    exponent = 0
    while a >= 2:
        a /= 2
        exponent += 1
    while a < 1:
        a *= 2
        exponent -= 1
    q = a * 128
    n, remainder = divmod(q.numerator, q.denominator)
    if 2 * remainder > q.denominator or (2 * remainder == q.denominator and n % 2):
        n += 1
    return sign * F(n, 128) * F(2) ** exponent

# 512 dimensions (7,17) repeated; rms = sqrt((49+289)/2) = 13.
row = [7, 17] * 256
assert sum(t*t for t in row) == 512*13*13
x, y = F(7, 13), F(17, 13)
# FP32-representable coefficients, near 0.6 and 0.8; exact arithmetic in this model.
c, s = F(5033165, 8388608), F(13421773, 16777216)
rot = lambda a, b: (c*a-s*b, s*a+c*b)
original = tuple(map(bf16_normal, rot(bf16_normal(x), bf16_normal(y))))
fused_skip = tuple(map(bf16_normal, rot(x, y)))
assert (bf16_normal(x), bf16_normal(y)) == (F(69,128), F(167,128))
assert original == (F(-23,32), F(155,128))
assert fused_skip == (F(-185,256), F(39,32))
print('norm_bf16:', bf16_normal(x), bf16_normal(y))
print('rotate_after_round:', *original)
print('rotate_without_intermediate_round:', *fused_skip)
