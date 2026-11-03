#!/usr/bin/env python3
"""Small exhaustive check of a conditional GF(2) output-information bound."""
from itertools import product

def rank_mod2(rows):
    basis = {}
    for row in rows:
        value = sum((x & 1) << j for j,x in enumerate(row))
        while value:
            pivot = value.bit_length()-1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)

for w in ([[1,0],[1,0]], [[2]]):
    n,k=len(w),len(w[0])
    assert all(len(row)==k for row in w)
    outputs={tuple(sum(x[i]*w[i][j] for i in range(n)) for j in range(k)) for x in product((0,1),repeat=n)}
    parity={tuple(v & 1 for v in y) for y in outputs}
    rank=rank_mod2(w)
    assert len(parity)==2**rank and len(outputs)>=len(parity)
    print({'shape':[n,k], 'gf2_rank':rank, 'distinct_exact_outputs':len(outputs), 'distinct_parity_outputs':len(parity)})
