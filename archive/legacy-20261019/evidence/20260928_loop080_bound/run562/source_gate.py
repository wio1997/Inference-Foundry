"""CPU-only source gate for the uninstalled Run562 segmented cohort draft."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch


SOURCE = Path(__file__).with_name("segmented_serving_candidate.py")


class FakeBase:
    def __init__(self, runtime, *, initial_output_counts, max_output_tokens):
        self.runtime = runtime
        self.config = runtime.config
        self.initial_output_counts = list(initial_output_counts)
        self.remaining = [max_output_tokens - count for count in initial_output_counts]
        self.max_cycles = 4
        self._progress_baseline = [0] * 12
        self._parked = [False] * 12

    def _committed_progress(self):
        return list(self.runtime.progress)

    def _park_completed(self, progress):
        for slot, count in enumerate(progress):
            if count >= self.remaining[slot]:
                self._parked[slot] = True
        # Reproduce the Host-mirror flush edge: the original loop still uses
        # the pre-parking progress snapshot for its cycle-end decision.
        if self.runtime.flush_on_park:
            self.runtime.progress = [1] * 12


class FakeRuntime:
    def __init__(self, *, all_complete_first=False, flush_on_park=True):
        self.config = types.SimpleNamespace(batch_size=12, target_tokens_per_request=1)
        self.state = types.SimpleNamespace(
            accepted_tokens=torch.empty((12, 1), dtype=torch.int32),
            num_sampled=torch.empty((12,), dtype=torch.int32),
        )
        self.progress = [0] * 12
        self.steps = 0
        self.flush_on_park = flush_on_park
        self.all_complete_first = all_complete_first
        self.invalidations = 0

    def step(self):
        self.steps += 1
        if self.steps == 1:
            counts = [1] * 12 if self.all_complete_first else [1] + [0] * 11
            self.progress = list(counts)
        elif self.steps == 2:
            counts = [0] + [1] * 11
            self.progress = [1] * 12
        else:
            raise AssertionError("unexpected extra Runtime.step")
        self.state.num_sampled = torch.tensor(counts, dtype=torch.int32)
        accepted = torch.tensor(
            [[100 + slot if counts[slot] else -1] for slot in range(12)],
            dtype=torch.int32,
        )
        return types.SimpleNamespace(
            acceptance=types.SimpleNamespace(sampled_token_ids=accepted)
        )

    def invalidate_scheduled_metadata(self):
        self.invalidations += 1


package = types.ModuleType("run562test")
package.__path__ = []
sys.modules["run562test"] = package
fixed = types.ModuleType("run562test.fixed_serving")
fixed.FixedCohortServing = FakeBase
fixed.FixedCohortOutput = lambda **fields: types.SimpleNamespace(**fields)
sys.modules[fixed.__name__] = fixed
spec = importlib.util.spec_from_file_location("run562test.segmented", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

original_all_reduce = torch.distributed.all_reduce
original_is_initialized = torch.distributed.is_initialized
original_get_world_size = torch.distributed.get_world_size
original_npu = getattr(torch, "npu", None)
torch.npu = types.SimpleNamespace(synchronize=lambda: None)
mode = {"kind": None, "digest_reductions": 0}


def fake_all_reduce(tensor, op):
    if op == torch.distributed.ReduceOp.MAX:
        if mode["kind"] == "park" and tensor.numel() == 5:
            tensor[-1] += 1
        if (mode["kind"] == "resume" and tensor.numel() == 5
                and int(tensor[2]) == 2):
            tensor[-1] += 1
        if mode["kind"] == "tokens" and tensor.numel() == 8:
            tensor[0] += 1
        if mode["kind"] == "trajectory" and tensor.numel() == 8:
            mode["digest_reductions"] += 1
            if mode["digest_reductions"] == 2:
                tensor[0] += 1


torch.distributed.all_reduce = fake_all_reduce
torch.distributed.is_initialized = lambda: True
torch.distributed.get_world_size = lambda: 8
try:
    ids = [f"request-{slot}" for slot in range(12)]
    runtime = FakeRuntime()
    cohort = module.SegmentedCohortServing(
        runtime, generation=7, request_ids=ids,
        initial_output_counts=[0] * 12, max_output_tokens=1,
    )
    segment = cohort.pause_after_first()
    assert segment.generation == 7 and segment.slots == (0,)
    assert segment.token_ids == ((100,),) and runtime.steps == 1
    assert cohort._last_decision_progress == [1] + [0] * 11
    assert runtime.progress == [1] * 12  # parking flushed other mirrors
    try:
        cohort.finish_after_pause(expected_generation=8)
        raise AssertionError("stale generation was accepted")
    except RuntimeError as exc:
        assert "stale continuation" in str(exc)
    result = cohort.finish_after_pause(expected_generation=7)
    assert runtime.steps == 2, "must preserve original pre-parking stop decision"
    assert result.cycles == 2 and result.generated_output_counts == [1] * 12
    assert result.token_ids == [[100 + slot] for slot in range(12)]
    assert cohort.trajectory["cycles"] == 2
    assert cohort.trajectory["counts"] == [[1] + [0] * 11, [0] + [1] * 11]
    assert len(cohort.trajectory["sha256"]) == 64
    assert len(segment.output_digest) == 64
    try:
        cohort.finish_after_pause(expected_generation=7)
        raise AssertionError("duplicate continuation was accepted")
    except RuntimeError as exc:
        assert "already resumed" in str(exc)

    all_done = module.SegmentedCohortServing(
        FakeRuntime(all_complete_first=True), generation=7, request_ids=ids,
        initial_output_counts=[0] * 12, max_output_tokens=1,
    )
    try:
        all_done.pause_after_first()
        raise AssertionError("non-early publication accepted")
    except RuntimeError as exc:
        assert "no unfinished old work" in str(exc)

    mode["kind"] = "park"
    divergent = module.SegmentedCohortServing(
        FakeRuntime(), generation=7, request_ids=ids,
        initial_output_counts=[0] * 12, max_output_tokens=1,
    )
    try:
        divergent.pause_after_first()
        raise AssertionError("rank-divergent branch accepted")
    except RuntimeError as exc:
        assert "park/decision disagreement" in str(exc)
    assert divergent.segment is None

    mode["kind"] = "tokens"
    divergent_tokens = module.SegmentedCohortServing(
        FakeRuntime(), generation=7, request_ids=ids,
        initial_output_counts=[0] * 12, max_output_tokens=1,
    )
    try:
        divergent_tokens.pause_after_first()
        raise AssertionError("rank-divergent tokens accepted")
    except RuntimeError as exc:
        assert "early output disagreement" in str(exc)
    assert divergent_tokens.segment is None

    mode["kind"] = "resume"
    divergent_resume = module.SegmentedCohortServing(
        FakeRuntime(), generation=7, request_ids=ids,
        initial_output_counts=[0] * 12, max_output_tokens=1,
    )
    divergent_resume.pause_after_first()
    try:
        divergent_resume.finish_after_pause(expected_generation=7)
        raise AssertionError("rank-divergent resume accepted")
    except RuntimeError as exc:
        assert "park/decision disagreement" in str(exc)

    mode["kind"] = "trajectory"
    mode["digest_reductions"] = 0
    divergent_trajectory = module.SegmentedCohortServing(
        FakeRuntime(), generation=7, request_ids=ids,
        initial_output_counts=[0] * 12, max_output_tokens=1,
    )
    divergent_trajectory.pause_after_first()
    try:
        divergent_trajectory.finish_after_pause(expected_generation=7)
        raise AssertionError("rank-divergent final trajectory accepted")
    except RuntimeError as exc:
        assert "logical trajectory disagreement" in str(exc)

    mode["kind"] = None
    torch.distributed.get_world_size = lambda: 1
    wrong_group = module.SegmentedCohortServing(
        FakeRuntime(), generation=7, request_ids=ids,
        initial_output_counts=[0] * 12, max_output_tokens=1,
    )
    try:
        wrong_group.pause_after_first()
        raise AssertionError("wrong group size accepted")
    except RuntimeError as exc:
        assert "initialized TP8 group" in str(exc)
    print("Run562 CPU source gate PASS: stop-decision, generation, group, all-done, rank-branch/token/final negatives")
finally:
    torch.distributed.all_reduce = original_all_reduce
    torch.distributed.is_initialized = original_is_initialized
    torch.distributed.get_world_size = original_get_world_size
    if original_npu is None:
        del torch.npu
    else:
        torch.npu = original_npu
