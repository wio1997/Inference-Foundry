#!/usr/bin/env python3
"""Finite arithmetic check of the documented causal CP metadata transformation.

This is an independent CPU reference, not an execution of vLLM or its kernels.
Does not prove historical runtime arguments or source/binary identity.
"""
from itertools import product

cases = rows = 0
for n in range(1, 4):
    for q in product(range(6), repeat=n):
        actual = sum(q)
        if not actual:
            continue
        ends = [0]
        for v in q:
            ends.append(ends[-1] + v)
        for computed in product(range(4), repeat=n):
            for padding in range(4):
                padded_input = actual + padding
                for world in range(1, 5):
                    per_rank = (padded_input + world - 1) // world
                    for rank in range(world):
                        s0, e0 = rank * per_rank, (rank + 1) * per_rank
                        e = min(e0, actual)
                        s = min(s0, e)
                        global_positions = [c + i for c, count in zip(computed, q) for i in range(count)]
                        local_positions = global_positions[s:e]
                        clamped = [min(max(x, s0), e0) for x in ends]
                        qsl = [0]
                        for a, b in zip(clamped, clamped[1:]):
                            qsl.append(qsl[-1] + b - a)
                        qsl = [min(x, e - s) for x in qsl]
                        for req in range(n):
                            q_local = qsl[req + 1] - qsl[req]
                            if not q_local:
                                continue
                            suffix = ends[req + 1] - min(max(ends[req + 1], s), e)
                            local_l = max(0, computed[req] + q[req] - suffix)
                            for i in range(q_local):
                                p = local_positions[qsl[req] + i]
                                assert p == local_l - q_local + i
                                rows += 1
                        cases += 1
print(f'checked {cases} causal configurations and {rows} real local rows')
# Noncausal retains the global L rather than removing a later-rank suffix.
computed, q, start, end = 100, 6, 2, 4
p, q_local, global_l = computed + start, end - start, computed + q
assert p != global_l - q_local
print(f'noncausal counterexample: p={p}, L-Q+i={global_l-q_local}, i=0')
