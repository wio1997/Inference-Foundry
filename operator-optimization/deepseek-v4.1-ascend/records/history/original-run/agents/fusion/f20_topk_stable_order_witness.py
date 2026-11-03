#!/usr/bin/env python3
"""Abstract stable-order witnesses; does not invoke CANN or prove float semantics."""
from itertools import product

# Global tie order is original position. A stable sort after block concatenation
# instead preserves concatenation order and can change the result.
original = [(1, 0), (1, 1), (1, 2)]
block_concat = [original[0], original[2], original[1]]
global_ids = [x[1] for x in sorted(original, key=lambda x: -x[0])[:2]]
wrong_ids = [x[1] for x in sorted(block_concat, key=lambda x: -x[0])[:2]]
assert global_ids == [0, 1] and wrong_ids == [0, 2]

blocks = [[0, 3], [1, 4], [2, 5]]
checks = 0
for scores in product(range(3), repeat=6):
    key = lambda i: (-scores[i], i)
    for k in range(1, 7):
        target = sorted(range(6), key=key)[:k]
        candidates = [i for block in blocks for i in sorted(block, key=key)[:k]]
        assert sorted(candidates, key=key)[:k] == target
        checks += 1
print('stable block concatenation:', wrong_ids, 'original-order result:', global_ids)
print('finite abstract score/k checks:', checks)
print('These are CPU integer examples, not deployed Sort/NaN/rounding verification.')
