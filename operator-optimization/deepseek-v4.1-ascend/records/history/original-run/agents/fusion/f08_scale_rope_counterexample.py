#!/usr/bin/env python3
"""Exact arithmetic counterexample for reordering RMS weight and one rotary pair.

This checks an abstract real-arithmetic pair, not deployed parameters, a kernel,
BF16 numerical behavior, or historical execution.
"""
from fractions import Fraction as F

def rotate(x, c, s):
    a, b = x
    return (c*a - s*b, s*a + c*b)

def scale(x, g):
    return tuple(a*b for a, b in zip(x, g))

x = (F(1), F(0))
g = (F(1), F(2))
c, s = F(3, 5), F(4, 5)
assert c*c + s*s == 1
before = rotate(scale(x, g), c, s)
after = scale(rotate(x, c, s), g)
assert before == (F(3, 5), F(4, 5))
assert after == (F(3, 5), F(8, 5))
assert before != after
# Equal weights commute with this pair rotation in exact arithmetic.
equal = (F(2), F(2))
assert rotate(scale(x, equal), c, s) == scale(rotate(x, c, s), equal)
print('R(Dx) =', tuple(map(str, before)))
print('D(Rx) =', tuple(map(str, after)))
print('Exact abstract pair only; common RMS denominator omitted; actual weights and finite precision not tested.')
