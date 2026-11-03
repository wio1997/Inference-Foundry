#!/usr/bin/env python3
"""CPU counterexamples to two C2 ring/pool fusion reorderings."""

start, used, ring_size = 0, 33, 32
reference = [None] * ring_size
tail_count = min(used, ring_size)
for token in range(used - tail_count, used):
    reference[(start + token) % ring_size] = token
assert reference[0] == 32

# A proposed naive parallel writer can complete the older colliding token last.
unordered = [None] * ring_size
for token in list(range(1, 33)) + [0]:
    unordered[(start + token) % ring_size] = token
assert unordered[0] == 0 and unordered[0] != reference[0]

# One incomplete 2-token group writes no pooled row; prior contents persist
# if a reused output buffer is not zeroed.
previous = [7.0]
groups = (0 + 1) // 2 - 0 // 2
assert groups == 0
reference_out = [0.0] * len(previous)
incorrect_without_zero = previous[:]
assert reference_out != incorrect_without_zero
print('ring slot 0: tail-filtered=', reference[0], 'unordered=', unordered[0])
print('incomplete group: zeroed=', reference_out, 'without initialization=', incorrect_without_zero)
