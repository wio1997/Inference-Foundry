"""Recompute the illustrative dense-MAC arithmetic; not a hardware certificate."""
import json
import math
from pathlib import Path

data = json.loads(Path(__file__).with_name('f16-conditional-example.json').read_text())
capacity = data['assumed_capacity_MAC_per_s']
assert capacity == 24 * 1_800_000_000 * 4096
for row in data['rows']:
    per_row = (4096 * 1024, 1024 * 5120)[row['projection'] - 1]
    work = row['physical_rows'] * per_row
    ideal_us = work / capacity * 1_000_000
    ratio = ideal_us / row['profiled_task_us'] * 100
    assert row['logical_MAC'] == work
    assert math.isclose(row['zero_credit_ideal_us'], ideal_us, rel_tol=1e-12)
    assert math.isclose(row['conditional_ratio_percent'], ratio, rel_tol=1e-12)
    print(f"M={row['physical_rows']} projection={row['projection']} "
          f"model={ideal_us:.3f} us profile={row['profiled_task_us']:.3f} us "
          f"ratio={ratio:.2f}%")
print('Conditional arithmetic only; capacity, valid work and zero credit are unproved.')
