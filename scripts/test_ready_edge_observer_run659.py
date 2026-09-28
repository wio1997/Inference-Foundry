#!/usr/bin/env python3
"""CPU-only positive and negative checks for the typed ready-edge packet."""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path


class Event:
    counter = 0
    def __init__(self, enable_timing):
        assert enable_timing
        self.number = None
    def record(self, stream):
        Event.counter += 1
        self.number = Event.counter
    def query(self):
        return self.number is not None
    def elapsed_time(self, other):
        return float(other.number - self.number)


class Stream:
    def __init__(self, native):
        self.npu_stream = native
        self.device = "npu:0"


class NPU:
    Event = Event
    active = None
    @classmethod
    def current_stream(cls, device=None):
        return cls.active


class Storage:
    def __init__(self, pointer):
        self.pointer = pointer
    def data_ptr(self):
        return self.pointer


class Tensor:
    def __init__(self, pointer, shape, version=1):
        self.pointer = pointer
        self.shape = shape
        self._version = version
        self.dtype = "torch.int32"
        self.device = "npu:0"
    def untyped_storage(self):
        return Storage(self.pointer)
    def data_ptr(self):
        return self.pointer
    def stride(self):
        return (self.shape[-1], 1) if len(self.shape) == 2 else (1,)


root = Path("/data/wio/Inference_Foundry/runtime")
pkg = types.ModuleType("runtime")
pkg.__path__ = [str(root)]
fake_torch = types.SimpleNamespace(npu=NPU(), Tensor=Tensor)
sys.modules["torch"] = fake_torch
sys.modules["runtime"] = pkg
for name in ("bound_observer", "ready_edge_observer"):
    spec = importlib.util.spec_from_file_location(f"runtime.{name}", root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"runtime.{name}"] = module
    spec.loader.exec_module(module)
recorder_type = sys.modules["runtime.ready_edge_observer"].ReadyEdgeRecorder
base = sys.modules["runtime.bound_observer"]
current = Stream(1)
NPU.active = current
r = recorder_type((63, 64, 65), current_stream=current)
for cycle in (63, 64, 65):
    draft = Tensor(100, (12, 7), cycle)
    target = Tensor(200, (96,), cycle)
    for label in base.CURRENT:
        r.mark(cycle, label)
        if label == "state_advance_after":
            r.observe(cycle, "after_state.last_sampled_tokens", Tensor(300, (12,)),
                      after_label=label)
            r.observe(cycle, "after_state.num_computed_tokens", Tensor(400, (12,)),
                      after_label=label)
        elif label == "proposer_after":
            r.observe(cycle, "after_proposer.next_draft", Tensor(500+cycle, (12, 7)),
                      after_label=label)
        elif label == "draft_commit_after":
            r.observe(cycle, "after_commit.draft_tokens", draft, after_label=label)
        elif label == "prepare_target":
            r.observe(cycle, "after_prepare.target_input_ids", target, after_label=label)
            r.observe(cycle, "after_prepare.target_positions", Tensor(600, (96,)),
                      after_label=label)
            r.observe(cycle, "after_prepare.target_slot_mapping", Tensor(700, (96,)),
                      after_label=label)
        elif label == "target_before":
            r.observe(cycle, "target_forward.target_input_ids", target, after_label=label)
    r.wait_start(cycle, cycle-1)
    r.wait_end(cycle)
identity = {
    "rank": 0, "cohort": 5, "req_ids": [str(i) for i in range(12)],
    "run_ts": "CPU", "time_namespace": "test",
    "runtime_basis": {
        "cycles": 300,
        "canonical_effective_staged_trajectory_sha256": "a"*64,
        "accepted_count_history_sha256": "b"*64,
        "raw_padded_token_history_sha256": "c"*64,
    },
    "product_output_sha256": "d"*64,
    "generated_output_counts": [1024]*12,
}
result = r.export_after_existing_drain(identity)
assert len(result["typed_rows"]) == 3*8
negatives = 0
for action in (
    lambda: r.observe(63, "after_commit.draft_tokens", Tensor(100, (12, 7)),
                      after_label="draft_commit_after"),
    lambda: r.observe(63, "bad", Tensor(42, (1,)), after_label="missing"),
):
    try:
        action()
    except RuntimeError:
        negatives += 1
    else:
        raise AssertionError("negative accepted")
# Reject a descriptor captured on the wrong stream and a changed consumer generation.
NPU.active = Stream(2)
try:
    r.observe(63, "drift", Tensor(44, (1,)), after_label="target_before")
except RuntimeError:
    negatives += 1
else:
    raise AssertionError("stream drift accepted")
NPU.active = current
read = next(row for row in r.typed_rows
            if row["cycle"] == 64 and row["role"] == "target_forward.target_input_ids")
read["python_version"] += 1
try:
    r.export_after_existing_drain(identity)
except RuntimeError:
    negatives += 1
else:
    raise AssertionError("changed input generation accepted")
print({"status": "pass", "typed_rows": len(result["typed_rows"]),
       "negative_cases": negatives})
