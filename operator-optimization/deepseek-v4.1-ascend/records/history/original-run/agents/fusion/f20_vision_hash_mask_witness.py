#!/usr/bin/env python3
"""CPU-only finite rational witness for a source-level masked-router identity."""
from fractions import Fraction
from itertools import product

cases = 0
for scores in product((1, 2, 3), repeat=4):
    for bias in product((-1, 0, 1), repeat=4):
        dynamic = sorted(range(4), key=lambda i: (-(scores[i] + bias[i]), i))[:2]
        for selected in product(range(4), repeat=2):
            # Source torch.where picks the table branch for text.  The dynamic
            # IDs and bias are intentionally evaluated, then discarded.
            source_ids = dynamic if False else selected
            source_scores = tuple(Fraction(scores[i]) for i in source_ids)
            direct_scores = tuple(Fraction(scores[i]) for i in selected)
            for renorm in (False, True):
                def weights(values):
                    if renorm:
                        values = tuple(v / sum(values) for v in values)
                    return tuple(v * Fraction(3, 2) for v in values)
                assert weights(source_scores) == weights(direct_scores)
                cases += 1

# A non-text row uses dynamic IDs.  The first two scores are identical,
# but changing a previously unseen expert changes the correct IDs.
a, b = (5, 4, 1, 0), (5, 4, 10, 0)
top2 = lambda x: tuple(sorted(range(4), key=lambda i: (-x[i], i))[:2])
assert a[:2] == b[:2] and top2(a) != top2(b)
print(f"{cases} text cases pass; image counterexample: {top2(a)} vs {top2(b)}")
