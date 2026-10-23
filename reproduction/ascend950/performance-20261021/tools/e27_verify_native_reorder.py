#!/usr/bin/env python3
"""CPU regression of the exact installed native reorder helper, not a rewrite."""
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
source = json.loads((ROOT / "evidence/E27-batch-order-source.json").read_text())
ns = {"np": np}
exec(source["native_reorder_function"], ns)
native = ns["reorder_batch_to_split_decodes_and_prefills"]


class Batch:
    def __init__(self, rows):
        self.req_ids = [r[0] for r in rows]
        self.num_computed_tokens_cpu = np.array([r[1] for r in rows])
        self.num_prompt_tokens = np.array([r[2] for r in rows])
        self.kv_row_marker = list(self.req_ids)

    def swap_states(self, i, j):
        for field in [self.req_ids, self.num_computed_tokens_cpu,
                      self.num_prompt_tokens, self.kv_row_marker]:
            field[i], field[j] = field[j], field[i]


class Output:
    pass


cases = [
    ("reported_mixed_batch", [("d0", 32800, 32768, 6), ("p", 32132, 32768, 636)] +
     [("d" + str(i), 32800, 32768, 6) for i in range(1, 6)]),
    ("short_prefill_not_a_decode", [("p1", 32767, 32768, 1),
                                    ("d", 32800, 32768, 6),
                                    ("p4", 32764, 32768, 4)]),
    ("fresh_prefill_before_decode", [("fresh", 0, 32768, 8192), ("d", 32800, 32768, 6)]),
    ("all_decode", [("d0", 32800, 32768, 6), ("d1", 32800, 32768, 1)]),
]
results = []
for name, rows in cases:
    batch = Batch(rows)
    expected = {r[0]: (r[1], r[2]) for r in rows}
    output = Output()
    output.num_scheduled_tokens = {r[0]: r[3] for r in rows}
    changed = native(batch, output, decode_threshold=11)
    prefilling = batch.num_computed_tokens_cpu < batch.num_prompt_tokens
    num_decodes = int((~prefilling).sum())
    assert not prefilling[:num_decodes].any(), name
    for i, rid in enumerate(batch.req_ids):
        assert expected[rid] == (batch.num_computed_tokens_cpu[i], batch.num_prompt_tokens[i]), name
        assert batch.kv_row_marker[i] == rid, name
    results.append(dict(case=name, passed=True, changed=bool(changed), order=batch.req_ids))
print(json.dumps(dict(passed=True, cases=results)))
