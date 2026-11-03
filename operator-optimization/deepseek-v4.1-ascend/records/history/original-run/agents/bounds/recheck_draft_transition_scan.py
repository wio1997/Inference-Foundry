#!/usr/bin/env python3
"""CPU check of finite-state transition composition, not a speed measurement."""
import json

# Each f maps state 0 or 1 to the next state. These are only a deterministic
# toy and are not DSpark weights, logits, random draws or a hardware trace.
functions = [(1, 0), (0, 0), (1, 0)]

def compose(after, before):
    return tuple(after[before[s]] for s in range(2))

# Hillis-Steele inclusive prefix: each round reads the previous snapshot.
# Each round can be evaluated in parallel in the stated ideal PRAM model;
# this Python implementation deliberately runs only on the CPU.
prefix = functions[:]
span = 1
rounds = 0
while span < len(prefix):
    old = prefix[:]
    prefix = [compose(old[i], old[i-span]) if i >= span else old[i]
              for i in range(len(old))]
    span *= 2
    rounds += 1
seed = 0
scan_outputs = [f[seed] for f in prefix]
sequential = []
state = seed
for f in functions:
    state = f[state]
    sequential.append(state)
assert scan_outputs == sequential == [1, 0, 1]
print(json.dumps({'K': len(functions), 'states': 2, 'seed': seed,
                  'inclusive_prefix_tables': prefix,
                  'rounds': rounds, 'scan_outputs': scan_outputs,
                  'sequential_outputs': sequential, 'device_test': False},
                 sort_keys=True))
