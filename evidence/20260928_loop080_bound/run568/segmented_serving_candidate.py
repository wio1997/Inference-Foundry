"""Private first-completion continuation gate for the frozen c12 Extreme path.

No slot is reused. The original twelve-request Runtime and its KV bindings
remain alive across the temporary return to EngineCore.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import time
from typing import Any, Sequence

import torch

from .fixed_serving import FixedCohortOutput, FixedCohortServing


@dataclass(frozen=True)
class EarlyCohortSegment:
    generation: int
    request_ids: tuple[str, ...]
    slots: tuple[int, ...]
    token_ids: tuple[tuple[int, ...], ...]
    cycle: int
    pause_start_ns: int
    output_digest: str


class SegmentedCohortServing(FixedCohortServing):
    """Pause once after an actual completed slot, then run the same cohort."""

    def __init__(self, *args, generation: int, request_ids: Sequence[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ids = tuple(str(rid) for rid in request_ids)
        if (generation < 0 or len(ids) != self.config.batch_size
                or any(not rid for rid in ids) or len(set(ids)) != len(ids)):
            raise ValueError("segmented cohort needs one generation and twelve unique IDs")
        self.generation = int(generation)
        self.request_ids = ids
        self._request_digest = int.from_bytes(
            hashlib.sha256("\0".join(ids).encode()).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        cfg = self.config
        device = self.runtime.state.accepted_tokens.device
        self._token_history = torch.empty(
            (self.max_cycles, cfg.batch_size, cfg.target_tokens_per_request),
            dtype=self.runtime.state.accepted_tokens.dtype,
            device=device,
        )
        self._count_history = torch.empty(
            (self.max_cycles, cfg.batch_size), dtype=torch.int32, device=device
        )
        self._cycles = 0
        self._last_decision_progress: list[int] | None = None
        self._segment: EarlyCohortSegment | None = None
        self._resume_ns: int | None = None
        self._trajectory: dict[str, Any] | None = None

    @property
    def segment(self) -> EarlyCohortSegment | None:
        return self._segment

    @property
    def resume_pause_ms(self) -> float | None:
        if self._segment is None or self._resume_ns is None:
            return None
        return (self._resume_ns - self._segment.pause_start_ns) / 1e6

    @property
    def trajectory(self) -> dict[str, Any] | None:
        """Full logical sampled-token/count ledger, available only after finish."""
        return self._trajectory

    @staticmethod
    def _digest_signature(hex_digest: str, device: torch.device) -> torch.Tensor:
        # Eight unsigned 32-bit words fit in signed int64, so MIN/MAX checks
        # the entire SHA256 digest rather than a truncated prefix.
        digest = bytes.fromhex(hex_digest)
        return torch.tensor(
            [int.from_bytes(digest[i:i + 4], "big") for i in range(0, 32, 4)],
            dtype=torch.int64,
            device=device,
        )

    def _advance_one(self) -> list[int]:
        if self._cycles >= self.max_cycles:
            raise RuntimeError("segmented cohort exceeded its cycle bound")
        result = self.runtime.step()
        index = self._cycles
        self._token_history[index].copy_(result.acceptance.sampled_token_ids)
        self._count_history[index].copy_(self.runtime.state.num_sampled)
        self._cycles += 1
        progress = self._committed_progress()
        self._park_completed(progress)
        # FixedCohortServing.run() decides termination from this pre-parking
        # snapshot. Parking may flush Host mirrors for other slots.
        self._last_decision_progress = progress
        return progress

    def _cohort_complete_at_decision(self) -> bool:
        progress = self._last_decision_progress
        if progress is None:
            raise RuntimeError("no Runtime step has been committed")
        return all(
            progress[slot] - self._progress_baseline[slot] >= self.remaining[slot]
            for slot in range(self.config.batch_size)
        )

    def _read_tokens(self, slots: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        width = self.config.target_tokens_per_request
        # Only the selected slots are copied to Host at the early boundary.
        tokens = self._token_history[: self._cycles, list(slots)].cpu()
        counts = self._count_history[: self._cycles, list(slots)].cpu()
        output: list[tuple[int, ...]] = []
        for col, slot in enumerate(slots):
            row_output: list[int] = []
            for cycle in range(self._cycles):
                count = int(counts[cycle, col])
                if count < 0 or count > width:
                    raise RuntimeError("acceptance count left fixed contract")
                row = tokens[cycle, col, :count].tolist()
                if any(value < 0 for value in row):
                    raise RuntimeError("accepted output contains padding")
                needed = self.remaining[slot] - len(row_output)
                if needed > 0:
                    row_output.extend(int(value) for value in row[:needed])
            if len(row_output) != self.remaining[slot]:
                raise RuntimeError("early output incomplete")
            output.append(tuple(row_output))
        return tuple(output)

    def _agreed_completed_slots(self) -> tuple[int, ...]:
        """Every rank enters this after each step, before a local pause branch."""
        slots = tuple(i for i, parked in enumerate(self._parked) if parked)
        signature = torch.tensor(
            [self.generation, self._request_digest, self._cycles,
             sum(1 << slot for slot in slots),
             int(self._cohort_complete_at_decision())],
            dtype=torch.int64,
            device=self.runtime.state.accepted_tokens.device,
        )
        minimum, maximum = signature.clone(), signature.clone()
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        if not bool(torch.equal(minimum, maximum)):
            raise RuntimeError("all-eight generation/cycle/park/decision disagreement")
        return slots

    @torch.inference_mode()
    def pause_after_first(self) -> EarlyCohortSegment:
        return self._pause_after_n(1)

    @torch.inference_mode()
    def pause_after_three(self) -> EarlyCohortSegment:
        return self._pause_after_n(3)

    @torch.inference_mode()
    def _pause_after_n(self, required: int) -> EarlyCohortSegment:
        if self._segment is not None:
            raise RuntimeError("cohort was already segmented")
        if (not torch.distributed.is_initialized()
                or torch.distributed.get_world_size() != 8):
            raise RuntimeError("segmented cohort requires one initialized TP8 group")
        if not 1 <= required <= self.config.batch_size:
            raise ValueError("invalid completion count")
        while self._cycles < self.max_cycles:
            self._advance_one()
            completed = self._agreed_completed_slots()
            if len(completed) >= required:
                if self._cohort_complete_at_decision():
                    raise RuntimeError("no unfinished old work remains at pause")
                slots = tuple(completed[:required])
                probe_dir = os.getenv("EXTREME_PAUSE_STATE_PROBE_DIR")
                before = None
                if probe_dir:
                    from diagnostics.loop080_pause_state_probe_run567 import capture
                    # The first fence completes all local streams before the
                    # image; capture completes its own asynchronous clones.
                    torch.npu.synchronize()
                    before = capture(self)
                # Drain Target/Draft side-stream users before the worker returns.
                self.runtime.invalidate_scheduled_metadata()
                torch.npu.synchronize()
                token_ids = self._read_tokens(slots)
                output_bytes = json.dumps(
                    {"generation": self.generation, "request_ids": self.request_ids,
                     "slots": slots, "cycle": self._cycles, "token_ids": token_ids},
                    separators=(",", ":"),
                ).encode()
                output_digest = hashlib.sha256(output_bytes).hexdigest()
                token_signature = self._digest_signature(
                    output_digest, self.runtime.state.accepted_tokens.device,
                )
                token_min, token_max = token_signature.clone(), token_signature.clone()
                torch.distributed.all_reduce(token_min, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(token_max, op=torch.distributed.ReduceOp.MAX)
                if not bool(torch.equal(token_min, token_max)):
                    raise RuntimeError("all-eight early output disagreement")
                self._segment = EarlyCohortSegment(
                    generation=self.generation, request_ids=self.request_ids,
                    slots=slots, token_ids=token_ids, cycle=self._cycles,
                    pause_start_ns=time.perf_counter_ns(),
                    output_digest=output_digest,
                )
                if before is not None:
                    from diagnostics.loop080_pause_state_probe_run567 import capture, compare
                    torch.npu.synchronize()
                    after = capture(self)
                    report = compare(before, after)
                    report.update(
                        rank=int(torch.distributed.get_rank()),
                        generation=self.generation, cycle=self._cycles,
                        slots=list(slots), request_ids=list(self.request_ids),
                        scope="captured pause-transition fields only; no full physical KV or continuation W0",
                    )
                    os.makedirs(probe_dir, exist_ok=True)
                    with open(os.path.join(probe_dir,
                                           f"pause_state_rank{report['rank']}_cohort{self.generation}.json"),
                              "w", encoding="utf-8") as handle:
                        json.dump(report, handle, separators=(",", ":"))
                return self._segment
        raise RuntimeError("required slots never completed")

    @torch.inference_mode()
    def finish_after_pause(self, *, expected_generation: int) -> FixedCohortOutput:
        if self._segment is None or self._resume_ns is not None:
            raise RuntimeError("no resumable segment or already resumed")
        if expected_generation != self.generation:
            raise RuntimeError("stale continuation generation")
        self._resume_ns = time.perf_counter_ns()
        while not self._cohort_complete_at_decision():
            self._advance_one()
            self._agreed_completed_slots()
        self.runtime.invalidate_scheduled_metadata()
        tokens_cpu = self._token_history[: self._cycles].cpu()
        counts_cpu = self._count_history[: self._cycles].cpu()
        width = self.config.target_tokens_per_request
        output: list[list[int]] = [[] for _ in range(self.config.batch_size)]
        staged = [0] * self.config.batch_size
        for cycle in range(self._cycles):
            for slot in range(self.config.batch_size):
                count = int(counts_cpu[cycle, slot])
                if count < 0 or count > width:
                    raise RuntimeError("acceptance count left fixed contract")
                row = tokens_cpu[cycle, slot, :count].tolist()
                if any(token < 0 for token in row):
                    raise RuntimeError("accepted output contains padding")
                staged[slot] += count
                needed = self.remaining[slot] - len(output[slot])
                if needed > 0:
                    output[slot].extend(int(token) for token in row[:needed])
        if [len(row) for row in output] != self.remaining:
            raise RuntimeError("segmented cohort final output incomplete")
        for slot, early in zip(self._segment.slots, self._segment.token_ids):
            if tuple(output[slot]) != early:
                raise RuntimeError("old slot changed after early publication")
        sampled: list[list[list[int]]] = []
        count_rows: list[list[int]] = []
        for cycle in range(self._cycles):
            cycle_counts: list[int] = []
            cycle_sampled: list[list[int]] = []
            for slot in range(self.config.batch_size):
                count = int(counts_cpu[cycle, slot])
                cycle_counts.append(count)
                cycle_sampled.append([
                    int(token) for token in tokens_cpu[cycle, slot, :count].tolist()
                ])
            count_rows.append(cycle_counts)
            sampled.append(cycle_sampled)
        trajectory_bytes = json.dumps(
            {"generation": self.generation, "request_ids": self.request_ids,
             "counts": count_rows, "sampled": sampled},
            separators=(",", ":"),
        ).encode()
        self._trajectory = {
            "generation": self.generation,
            "request_ids": self.request_ids,
            "cycles": self._cycles,
            "counts": count_rows,
            "sampled": sampled,
            "sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
        }
        trajectory_signature = self._digest_signature(
            self._trajectory["sha256"], self.runtime.state.accepted_tokens.device,
        )
        trajectory_min = trajectory_signature.clone()
        trajectory_max = trajectory_signature.clone()
        torch.distributed.all_reduce(trajectory_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(trajectory_max, op=torch.distributed.ReduceOp.MAX)
        if not bool(torch.equal(trajectory_min, trajectory_max)):
            raise RuntimeError("all-eight logical trajectory disagreement")
        means = {}
        for start, end in ((0, 8), (8, 64), (64, 128), (128, 192), (192, 256),
                           (256, 512), (512, 768), (768, 1024)):
            if start >= self._cycles:
                continue
            stop = min(end, self._cycles)
            means[f"{start}-{stop - 1}"] = float(
                counts_cpu[start:stop].sum().item()
                / ((stop - start) * self.config.batch_size)
            )
        return FixedCohortOutput(
            token_ids=output, cycles=self._cycles,
            initial_output_counts=list(self.initial_output_counts),
            requested_output_counts=list(self.remaining),
            generated_output_counts=[len(row) for row in output],
            staged_output_counts=staged,
            overshoot_tokens=sum(staged) - sum(len(row) for row in output),
            acceptance_window_means=means,
        )
