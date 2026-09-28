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
from types import SimpleNamespace
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
        protected_kv_tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self.config = config
        self.state = state
        self.target = target
        self.acceptance = acceptance
        self.proposer = proposer
        self.target_metadata = target_metadata
        self.kv_slot_audit = kv_slot_audit
        self.target_page_audit = target_page_audit
        self._protected_kv_tensors = protected_kv_tensors
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
        self._schedule_mode = os.getenv("EXTREME_SCHEDULE_NEXT_TARGET_METADATA", "off")
        if self._schedule_mode not in ("off", "serial", "overlap"):
            raise ValueError("invalid next-target metadata scheduling mode")
        if self._schedule_mode != "off" and target_metadata is None:
            raise ValueError("scheduled target metadata requires the native updater")
        self._schedule_verify = os.getenv("EXTREME_SCHEDULE_VERIFY") == "1"
        self._schedule_verify_count = 0
        self._schedule_pending = False
        self._schedule_launches = 0
        self._schedule_commits = 0
        self._schedule_invalidations = 0
        self._schedule_fallbacks = 0
        self._schedule_scratch = None
        self._schedule_updater = None
        self._schedule_stream = None
        self._schedule_event = None
        if self._schedule_mode != "off":
            self._schedule_scratch = SimpleNamespace(
                target_positions=torch.empty_like(state.target_positions),
                target_seq_lens=torch.empty_like(state.target_seq_lens),
                target_slot_mapping=torch.empty_like(state.target_slot_mapping),
            )
            self._schedule_updater = target_metadata.make_scratch()
            if self._schedule_mode == "overlap":
                self._schedule_stream = torch.npu.Stream(device=state.target_positions.device)
                self._schedule_event = torch.npu.Event()
        # Reuse the proven fixed-buffer preparation and state transition, not
        # its older bundled operator dispatch.
        self._state_machine = FixedDecodeRuntime(
            config,
            state,
            _UnreachableBundledOperators(),
        )
        if self._schedule_mode != "off":
            self._audit_scheduled_storage()

    def _scope(self, name: str):
        if self._profile_scopes:
            return record_function(name)
        return nullcontext()

    def _audit_scheduled_storage(self) -> None:
        """Prove private write intervals cannot touch live Target/DSpark/KV."""
        from .storage_intervals import audit_disjoint
        private = {}
        protected = {}
        def put(table, name, value):
            if torch.is_tensor(value):
                table[name] = value
        scratch = self._schedule_scratch
        for name in ("target_positions", "target_seq_lens", "target_slot_mapping"):
            put(private, f"scratch.{name}", getattr(scratch, name))
        updater = self._schedule_updater
        for name in ("start_pos", "local_seq_lens"):
            put(private, f"scratch_updater.{name}", getattr(updater, name))
        for index, binding in enumerate(updater.rotary):
            for name in ("target_cos", "target_sin", "local_cos", "local_sin"):
                put(private, f"scratch_rotary{index}.{name}", getattr(binding, name))
        for index, group in enumerate(updater.groups):
            for name in ("seq_lens", "input_positions", "start_pos",
                         "local_query_start_loc", "local_seq_lens",
                         "sas_metadata", "qli_metadata", "swa_slot_mapping"):
                put(private, f"scratch_group{index}.{name}", getattr(group, name))
        for name, value in vars(self.state).items():
            put(protected, f"state.{name}", value)
        for name in ("start_pos", "local_seq_lens"):
            put(protected, f"active_updater.{name}", getattr(self.target_metadata, name))
        for index, binding in enumerate(self.target_metadata.rotary):
            for name in ("target_cos", "target_sin", "local_cos", "local_sin"):
                put(protected, f"active_rotary{index}.{name}", getattr(binding, name))
        for index, group in enumerate(self.target_metadata.groups):
            for name in ("seq_lens", "input_positions", "start_pos",
                         "local_query_start_loc", "local_seq_lens",
                         "sas_metadata", "qli_metadata", "swa_slot_mapping"):
                put(protected, f"active_group{index}.{name}", getattr(group, name))
        common = getattr(self.proposer, "common_attn_metadata", None)
        if common is not None:
            for name in ("seq_lens", "slot_mapping", "positions", "query_start_loc",
                         "block_table_tensor"):
                put(protected, f"dspark_common.{name}", getattr(common, name, None))
        for index, tensor in enumerate(self._protected_kv_tensors):
            put(protected, f"target_kv{index}", tensor)
        report = audit_disjoint(private, protected)
        report["rank"] = (torch.distributed.get_rank()
                          if torch.distributed.is_initialized() else None)
        path = os.getenv("EXTREME_SCHEDULE_ALIAS_DIR")
        if path:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"rank{report['rank']}.jsonl"), "a",
                      encoding="utf-8") as handle:
                handle.write(json.dumps(report) + "\n")
        if not report["pass_"]:
            raise RuntimeError(f"target metadata scratch aliases live storage: "
                               f"{report['possible_aliases']}")

    def _calculate_next_target_metadata(self) -> None:
        scratch = self._schedule_scratch
        self._state_machine.prepare_target_geometry_to(
            scratch.target_positions, scratch.target_seq_lens,
            scratch.target_slot_mapping,
        )
        self._schedule_updater.update(scratch)

    def _launch_next_target_metadata(self) -> None:
        if self._schedule_pending:
            raise RuntimeError("previous target metadata scratch not consumed")
        if self._schedule_mode == "overlap":
            current = torch.npu.current_stream(self.state.target_positions.device)
            self._schedule_stream.wait_stream(current)
            with torch.npu.stream(self._schedule_stream):
                with self._scope("extreme::next_target_private_metadata"):
                    self._calculate_next_target_metadata()
                self._schedule_event.record()
        else:
            with self._scope("extreme::next_target_private_metadata"):
                self._calculate_next_target_metadata()
        self._schedule_pending = True
        self._schedule_launches += 1

    def _commit_next_target_metadata(self) -> None:
        if not self._schedule_pending:
            raise RuntimeError("no target metadata scratch to commit")
        if self._schedule_mode == "overlap":
            torch.npu.current_stream(self.state.target_positions.device).wait_event(
                self._schedule_event)
        scratch = self._schedule_scratch
        self.state.target_positions.copy_(scratch.target_positions)
        self.state.target_seq_lens.copy_(scratch.target_seq_lens)
        self.state.target_slot_mapping.copy_(scratch.target_slot_mapping)
        self.target_metadata.commit_from(self._schedule_updater)
        self._state_machine.prepare_target_ids()
        self._schedule_pending = False
        self._schedule_commits += 1

    def _verify_scheduled_target(self) -> None:
        """Compare committed scratch against a same-state serial rebuild."""
        if not self._schedule_verify:
            return
        active = self.target_metadata
        tensors = {
            "ids": self.state.target_input_ids,
            "positions": self.state.target_positions,
            "seq_lens": self.state.target_seq_lens,
            "slots": self.state.target_slot_mapping,
        }
        for index, binding in enumerate(active.rotary):
            tensors[f"rotary{index}.cos"] = binding.target_cos[:96]
            tensors[f"rotary{index}.sin"] = binding.target_sin[:96]
            if binding.local_cos is not None:
                tensors[f"rotary{index}.local_cos"] = binding.local_cos
                tensors[f"rotary{index}.local_sin"] = binding.local_sin
        for index, group in enumerate(active.groups):
            for name, count in (("seq_lens",12), ("input_positions",96),
                                ("start_pos",12), ("local_query_start_loc",13),
                                ("local_seq_lens",12), ("sas_metadata",97),
                                ("qli_metadata",25), ("swa_slot_mapping",96)):
                value = getattr(group, name)
                if value is not None:
                    tensors[f"group{index}.{name}"] = value[:count]
        snapshots = {name: value.clone() for name, value in tensors.items()}
        self._state_machine.prepare_target_inputs()
        active.update(self.state)
        mismatches = [name for name, value in tensors.items()
                      if not torch.equal(snapshots[name], value)]
        if mismatches:
            raise AssertionError(
                f"scheduled target metadata diverged at cycle {self.state.cycle_index}: "
                + ", ".join(mismatches))
        # The diagnostic must not replace the candidate with its reference:
        # restore the entire scratch result, including nondeterministic tails,
        # so the following target Graph consumes the candidate buffers.
        scratch = self._schedule_scratch
        self.state.target_positions.copy_(scratch.target_positions)
        self.state.target_seq_lens.copy_(scratch.target_seq_lens)
        self.state.target_slot_mapping.copy_(scratch.target_slot_mapping)
        active.commit_from(self._schedule_updater)
        self._state_machine.prepare_target_ids()
        self._schedule_verify_count += 1
        if self._schedule_verify_count in (1, 64, 128, 256):
            print(f"EXTREME_SCHEDULE_VERIFY rank={torch.distributed.get_rank()} "
                  f"cycles={self._schedule_verify_count} pass=1", flush=True)

    def invalidate_scheduled_metadata(self) -> None:
        """A serving parking change makes previously computed geometry stale."""
        if not self._schedule_pending:
            return
        if self._schedule_mode == "overlap":
            self._schedule_event.synchronize()
        self._schedule_pending = False
        self._schedule_invalidations += 1

    def schedule_lifetime(self) -> dict[str, int]:
        return {
            "launches": self._schedule_launches,
            "commits": self._schedule_commits,
            "invalidations": self._schedule_invalidations,
            "fallbacks": self._schedule_fallbacks,
        }

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
        profile_kwargs = {}
        if os.getenv("EXTREME_RUNTIME_CYCLE_PROFILE_MEMORY_ACCESS") == "1":
            from torch_npu.profiler import (
                AiCMetrics, ProfilerLevel, _ExperimentalConfig,
            )
            profile_kwargs["experimental_config"] = _ExperimentalConfig(
                profiler_level=ProfilerLevel.Level1,
                aic_metrics=AiCMetrics.MemoryAccess,
            )
        self._cycle_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            on_trace_ready=tensorboard_trace_handler(
                dir_name=self._cycle_profile_dir,
                worker_name=f"rank{rank}", analyse_flag=False,
            ),
            record_shapes=False, profile_memory=False, with_stack=False,
            **profile_kwargs,
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
            use_scheduled = self._schedule_pending
            if self._schedule_mode != "off" and not use_scheduled:
                self._schedule_fallbacks += 1
            with self._scope("extreme::prepare_target"):
                if use_scheduled:
                    self._commit_next_target_metadata()
                else:
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
                if not use_scheduled:
                    with self._scope("extreme::derived_target_metadata"):
                        self.target_metadata.update(self.state)
                mark("derived_target_metadata")
                if use_scheduled:
                    self._verify_scheduled_target()
            if self.kv_slot_audit is not None:
                self.kv_slot_audit.observe(
                    self.state.cycle_index, self.state.target_positions
                )
            if self.target_page_audit is not None:
                self.target_page_audit.observe(self.state.cycle_index, self.state)
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
            if self._schedule_mode == "overlap":
                self._launch_next_target_metadata()
            with self._scope("extreme::proposer"):
                next_draft = self.proposer.execute(
                    self.state,
                    target_output,
                    acceptance_output,
                )
            mark("proposer")
            if self._schedule_mode == "serial":
                self._launch_next_target_metadata()
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
