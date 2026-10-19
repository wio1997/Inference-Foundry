"""Definition-only arithmetic check, not a frontend/event or framework replay.

Run with Python 3.12 and the SHA-pinned downloaded stats.py as argv[1].
No imports from vLLM, no injected modules, and no edited function bodies.
The late scheduled timestamp is an explicit fixture, not a replayed event.
"""

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


path = Path(sys.argv[1])
raw = path.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
assert sha == "af1983ddc32fa4d3284e72212c6525daf961dd84429023a02d612b68b3112cf3"
tree = ast.parse(raw)
classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
methods = {"_time_since", "update_from_output", "update_from_finished_request"}
iteration = classes["IterationStats"]
iteration.body = [n for n in iteration.body if isinstance(n, ast.FunctionDef) and n.name in methods]
module = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    classes["RequestStateStats"], classes["FinishedRequestStats"], iteration,
], type_ignores=[])
ns = {"dataclass": dataclass, "envs": SimpleNamespace(VLLM_COMPUTE_NANS_IN_LOGITS=False)}
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)


def check(first_ts, scheduled_ts):
    stats = object.__new__(ns["IterationStats"])
    stats.iteration_timestamp = 98.0
    stats.num_generation_tokens = 0
    stats.time_to_first_tokens_iter = []
    stats.inter_token_latencies_iter = []
    stats.finished_requests = []
    req = ns["RequestStateStats"](arrival_time=97.0, queued_ts=9.0)
    output = SimpleNamespace(new_token_ids=[20], prefill_stats=None, events=None)
    stats.update_from_output(output, first_ts, True, req, None, None)
    # Fixture: the later native scheduling event has now set this field.
    req.scheduled_ts = scheduled_ts
    stats.iteration_timestamp = 100.0
    output.new_token_ids = list(range(63))
    stats.update_from_output(output, 12.0, False, req, None, None)
    stats.update_from_finished_request("length", "fixture", 256, 64, req)
    result = asdict(stats.finished_requests[0])
    assert result["num_generation_tokens"] == 64
    assert result["prefill_time"] == first_ts - scheduled_ts
    return result


early = check(10.0, 10.25)
normal = check(10.5, 10.25)
assert early["prefill_time"] == -0.25
assert normal["prefill_time"] == 0.25
print(json.dumps({
    "source_sha256": sha,
    "source_grade": "fixed_commit_reference_only_no_image_sha_match",
    "full_native_import": False,
    "executed": "Unmodified dataclass definitions and three original IterationStats methods",
    "fixtures": "object.__new__ bypasses IterationStats constructor; SimpleNamespace output/envs; no events; scheduled_ts assigned explicitly",
    "limits": "Synthetic timestamps; no actual OutputProcessor, EngineCore event replay, metrics exporter, torch, model or phase fix executed",
    "early_publication": early,
    "ordinary_order": normal,
}, indent=2))
