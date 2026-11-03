#!/usr/bin/env python3
"""CPU-only exhaustive examples of the restricted F23 semantic statement."""
from itertools import product
from fractions import Fraction
from pathlib import Path
import hashlib, json

scale = Fraction(3, 2)
records = []
for n in range(1, 5):
    values = list(product((0, 1), repeat=n))
    result = lambda x, y: tuple(scale * a + b for a, b in zip(x, y))
    zeros = (0,) * n
    outs = {result(zeros, y) for y in values}
    assert len(outs) == 2 ** n
    for i in range(n):
        changed = tuple(int(j == i) for j in range(n))
        assert result(zeros, zeros) != result(changed, zeros)
        assert result(zeros, zeros) != result(zeros, changed)
    records.append({'n': n, 'input_vectors': len(values), 'distinct_outputs_when_x_zero': len(outs),
                    'x_and_y_each_coordinate_has_distinguishing_pair': True,
                    'binary_strings_of_length_less_than_n': 2 ** n - 1})
print(json.dumps({'scale':str(scale),'records':records,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'limitations':['Finite examples check arithmetic only; general proof is in F23-COMMON-DEPENDENCY.md.',
                                 'This does not inspect live 169 tensors or establish a hardware communication cut.']},indent=2))
