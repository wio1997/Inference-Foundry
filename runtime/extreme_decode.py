"""Product-owned continuous decode driver for DeepSeek Extreme.

This module is the runtime boundary.  It does not import vLLM and it does not
accept scheduler, request, batch, or metadata-builder objects.  A bootstrap
layer may lend it loaded operator callables and physical cache tensors once;
after that, this driver owns every decode-cycle transition.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.profiler import record_function

from .fixed_decode import (
    AcceptanceOutput,
    FixedDecodeConfig,
    FixedDecodeRuntime,
    FixedDecodeState,
    TargetOutput,
)
from .target_adapter import FixedTargetAdapter


class AcceptanceOperator(Protocol):
    def execute(
        self,
        state: FixedDecodeState,
        target: TargetOutput,
    ) -> AcceptanceOutput: ...


class TargetMetadataOperator(Protocol):
    def update(self, state: FixedDecodeState) -> None: ...


class ProposerOperator(Protocol):
    def execute(
        self,
        state: FixedDecodeState,
        target: TargetOutput,
        acceptance: AcceptanceOutput,
    ) -> torch.Tensor: ...

    def state_fingerprint(self) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class CycleResult:
    cycle: int
    acceptance: AcceptanceOutput
    target_state: dict[str, torch.Tensor]
    proposer_state: dict[str, torch.Tensor]


class ExtremeDecodeRuntime:
    """Own the complete fixed decode execution order and mutable state."""

    stage_order = (
        "prepare_target",
        "target",
        "acceptance",
        "state_advance",
        "proposer",
    )

    def __init__(
        self,
        config: FixedDecodeConfig,
        state: FixedDecodeState,
        target: FixedTargetAdapter,
        acceptance: AcceptanceOperator,
        proposer: ProposerOperator,
        target_metadata: TargetMetadataOperator | None = None,
        kv_slot_audit=None,
        target_page_audit=None,
    ) -> None:
        self.config = config
        self.state = state
        self.target = target
        self.acceptance = acceptance
        self.proposer = proposer
        self.target_metadata = target_metadata
        self.kv_slot_audit = kv_slot_audit
        self.target_page_audit = target_page_audit
        if target_metadata is not None:
            self.stage_order = (
                "prepare_target", "derived_target_metadata", "target",
                "acceptance", "state_advance", "proposer",
            )
        self._profile_scopes = os.getenv("EXTREME_RUNTIME_PROFILE_SCOPES") == "1"
        self._diagnose = os.getenv("EXTREME_RUNTIME_DIAGNOSE") == "1"
        self._diagnose_limit = int(os.getenv("EXTREME_RUNTIME_TRACE_CYCLES", "1024"))
        self._profile_dag = os.getenv("EXTREME_RUNTIME_PROFILE_DAG") == "1"
        self._cycle_profile_dir = os.getenv("EXTREME_RUNTIME_CYCLE_PROFILE_DIR")
        self._cycle_profile_start = int(os.getenv(
            "EXTREME_RUNTIME_CYCLE_PROFILE_START", "64"))
        self._cycle_profile_count = int(os.getenv(
            "EXTREME_RUNTIME_CYCLE_PROFILE_COUNT", "2"))
        self._cycle_profile_sync_target = (
            os.getenv("EXTREME_RUNTIME_CYCLE_PROFILE_SYNC_TARGET") == "1")
        if self._cycle_profile_dir and (
            self._cycle_profile_start < 0 or self._cycle_profile_count < 1
        ):
            raise ValueError("invalid Extreme cycle profiler window")
        self._cycle_profiler = None
        self._cycle_profile_rank = None
        self._cycle_profile_started_ns = None
        self.diagnostic_cycles = []
        self.diagnostic_events = []
        # Reuse the proven fixed-buffer preparation and state transition, not
        # its older bundled operator dispatch.
        self._state_machine = FixedDecodeRuntime(
            config,
            state,
            _UnreachableBundledOperators(),
        )

    def _scope(self, name: str):
        if self._profile_scopes:
            return record_function(name)
        return nullcontext()

    def _profile_cycle_begin(self) -> None:
        if (not self._cycle_profile_dir or
                self.state.cycle_index != self._cycle_profile_start):
            return
        from torch_npu.profiler import (
            ProfilerActivity, profile, tensorboard_trace_handler,
        )
        rank = (torch.distributed.get_rank()
                if torch.distributed.is_initialized() else os.getpid())
        self._cycle_profile_rank = rank
        os.makedirs(self._cycle_profile_dir, exist_ok=True)
        self._cycle_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            on_trace_ready=tensorboard_trace_handler(
                dir_name=self._cycle_profile_dir,
                worker_name=f"rank{rank}", analyse_flag=False,
            ),
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        self._cycle_profile_started_ns = time.time_ns()
        self._cycle_profiler.start()

    def _profile_cycle_end(self) -> None:
        if (self._cycle_profiler is None or
                self.state.cycle_index !=
                self._cycle_profile_start + self._cycle_profile_count):
            return
        self._cycle_profiler.stop()
        path = os.path.join(
            self._cycle_profile_dir,
            f"rank{self._cycle_profile_rank}_window.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "rank": self._cycle_profile_rank,
                "first_cycle": self._cycle_profile_start,
                "cycle_count": self._cycle_profile_count,
                "started_ns": self._cycle_profile_started_ns,
                "stopped_ns": time.time_ns(),
            }, handle, indent=2)
        self._cycle_profiler = None

    @torch.inference_mode()
    def step(self) -> CycleResult:
        self._profile_cycle_begin()
        diag = (
            {} if self._diagnose and len(self.diagnostic_cycles) < self._diagnose_limit
            else None
        )
        markers = []
        def mark(label):
            if diag is not None or self._profile_dag:
                event = torch.npu.Event(enable_timing=True)
                event.record()
                markers.append((label, event))
        with self._scope("extreme::cycle"):
            if diag is not None:
                diag["num_computed_before"] = self.state.num_computed_tokens.clone()
                diag["last_token_before"] = self.state.last_sampled_tokens.clone()
                diag["draft_before"] = self.state.draft_tokens.clone()
            mark("begin")
            with self._scope("extreme::prepare_target"):
                self._state_machine.prepare_target_inputs()
            mark("prepare_target")
            if diag is not None:
                diag["target_input_ids"] = self.state.target_input_ids.view(
                    self.config.batch_size, self.config.target_tokens_per_request
                ).clone()
                diag["target_positions"] = self.state.target_positions.view(
                    self.config.batch_size, self.config.target_tokens_per_request
                ).clone()
            if self.target_metadata is not None:
                with self._scope("extreme::derived_target_metadata"):
                    self.target_metadata.update(self.state)
                mark("derived_target_metadata")
            if self.kv_slot_audit is not None:
                self.kv_slot_audit.observe(
                    self.state.cycle_index, self.state.target_positions
                )
            if self.target_page_audit is not None:
                self.target_page_audit.observe(self.state.cycle_index)
            if (self.target_page_audit is not None and
                    self.target_page_audit.self_replay and
                    self.state.cycle_index < self.target_page_audit.limit):
                with self._scope("extreme::target_self_replay"):
                    target_output, acceptance_output = (
                        self.target_page_audit.execute_with_self_replay(
                            self.target, self.state, self.acceptance
                        )
                    )
                mark("target")
            else:
                with self._scope("extreme::target"):
                    if self._cycle_profiler and self._cycle_profile_sync_target:
                        torch.npu.synchronize()
                    target_output = self.target.execute(self.state)
                    if self._cycle_profiler and self._cycle_profile_sync_target:
                        torch.npu.synchronize()
                mark("target")
                with self._scope("extreme::acceptance"):
                    acceptance_output = self.acceptance.execute(
                        self.state, target_output
                    )
            if diag is not None:
                diag["target_argmax"] = target_output.logits.argmax(dim=-1).view(
                    self.config.batch_size, self.config.target_tokens_per_request
                ).clone()
            mark("acceptance")
            if diag is not None:
                diag["accepted"] = acceptance_output.sampled_token_ids.clone()
                diag["counts"] = acceptance_output.num_sampled.clone()
            with self._scope("extreme::state_advance"):
                self._state_machine.advance_state(acceptance_output)
            mark("state_advance")
            with self._scope("extreme::proposer"):
                next_draft = self.proposer.execute(
                    self.state,
                    target_output,
                    acceptance_output,
                )
            mark("proposer")
            if self.kv_slot_audit is not None:
                self.kv_slot_audit.observe_dspark_context(
                    self.state.cycle_index,
                    self.state.target_positions,
                    self.proposer.proposer,
                )
            if diag is not None:
                diag["next_draft"] = next_draft.clone()
            with self._scope("extreme::draft_commit"):
                if tuple(next_draft.shape) != tuple(
                    self.state.draft_tokens.shape
                ):
                    raise ValueError("DSpark draft tensor shape changed")
                self.state.draft_tokens.copy_(next_draft)
                self.state.cycle_index += 1
            mark("draft_commit")
            if diag is not None:
                self.diagnostic_cycles.append(diag)
            if markers:
                self.diagnostic_events.append(markers)
            self._profile_cycle_end()
            return CycleResult(
                cycle=self.state.cycle_index,
                acceptance=acceptance_output,
                target_state=self.target.state_fingerprint(),
                proposer_state=self.proposer.state_fingerprint(),
            )

    @torch.inference_mode()
    def run(self, cycles: int) -> list[CycleResult]:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        return [self.step() for _ in range(cycles)]


class _UnreachableBundledOperators:
    """Guard against accidentally falling back to the legacy bundled path."""

    def __getattr__(self, name: str):
        raise RuntimeError(f"bundled operator dispatch is forbidden: {name}")
