"""Fixed-cohort serving shell for the Extreme decode runtime.

The shell keeps a 12-request cohort inside the product-owned decode loop until
every request has enough tokens to satisfy its serving limit. Accepted tokens
are staged in fixed device buffers and copied to the host once, after decode;
the serving control plane is not re-entered between cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import torch

from .extreme_decode import ExtremeDecodeRuntime


@dataclass(frozen=True)
class FixedCohortOutput:
    token_ids: list[list[int]]
    cycles: int
    initial_output_counts: list[int]
    requested_output_counts: list[int]
    generated_output_counts: list[int]
    staged_output_counts: list[int]
    overshoot_tokens: int
    acceptance_window_means: dict[str, float]


class FixedCohortServing:
    """Run one fixed-shape cohort without per-cycle Host orchestration."""

    def __init__(
        self,
        runtime: ExtremeDecodeRuntime,
        *,
        initial_output_counts: Sequence[int],
        max_output_tokens: int,
    ) -> None:
        self.runtime = runtime
        self.config = runtime.config
        initial = [int(value) for value in initial_output_counts]
        if len(initial) != self.config.batch_size:
            raise ValueError("initial output count must have one value per slot")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        remaining = [max_output_tokens - value for value in initial]
        if any(value <= 0 for value in remaining):
            raise ValueError("cohort must enter Extreme before its output limit")
        self.initial_output_counts = initial
        self.remaining = remaining
        self._progress_baseline = self._committed_progress()
        self._initial_positions = [
            int(value) for value in runtime.state.num_computed_tokens.cpu().tolist()
        ]
        self._parked = [False] * self.config.batch_size
        # The final valid target may cross the exact output boundary by seven
        # positions. Serving must reserve at least two extra KV blocks.
        if os.getenv("EXTREME_RUNTIME_SERVE") != "1":
            raise ValueError("fixed serving requires scheduler KV reservation")
        reserve = int(os.getenv("EXTREME_RUNTIME_RESERVE_TOKENS", "0"))
        if reserve < max(remaining) + 2 * self.config.block_size:
            raise ValueError("fixed serving requires output limit plus two KV blocks")
        # Greedy verification emits at least one token per live slot per cycle.
        # The DSpark Host progress mirror trails by one cycle, hence +1.
        self.max_cycles = max(remaining) + 1

    def _committed_progress(self) -> list[int]:
        provider = getattr(
            self.runtime.proposer, "committed_emitted_token_count", None
        )
        if provider is not None:
            return [int(value) for value in provider().tolist()]
        if self.runtime.state.emitted_token_count.device.type != "cpu":
            raise RuntimeError(
                "NPU serving requires the proposer Host progress mirror"
            )
        return [
            int(value)
            for value in self.runtime.state.emitted_token_count.tolist()
        ]

    def _park_completed(self, progress: list[int]) -> None:
        slots = [
            slot for slot in range(self.config.batch_size)
            if not self._parked[slot]
            and progress[slot] - self._progress_baseline[slot]
            >= self.remaining[slot]
        ]
        if not slots:
            return
        # Fixed shape requires parked slots to keep executing. Rewind each to
        # its own reserved KV blocks and freeze its logical output progress.
        positions = [
            self._initial_positions[slot] + max(
                0, self.remaining[slot] - 2 * self.config.block_size
            )
            for slot in slots
        ]
        self.runtime.proposer.park_completed_slots(slots, positions)
        device = self.runtime.state.num_computed_tokens.device
        idx = torch.tensor(slots, dtype=torch.long, device=device)
        values = torch.tensor(
            positions, dtype=self.runtime.state.num_computed_tokens.dtype, device=device
        )
        self.runtime.state.num_computed_tokens[idx] = values
        self.runtime.state.active_mask[idx] = False
        for slot in slots:
            self._parked[slot] = True

    @torch.inference_mode()
    def run(self) -> FixedCohortOutput:
        cfg = self.config
        device = self.runtime.state.accepted_tokens.device
        width = cfg.target_tokens_per_request
        token_history = torch.empty(
            (self.max_cycles, cfg.batch_size, width),
            dtype=self.runtime.state.accepted_tokens.dtype,
            device=device,
        )
        count_history = torch.empty(
            (self.max_cycles, cfg.batch_size),
            dtype=torch.int32,
            device=device,
        )

        cycles = 0
        for index in range(self.max_cycles):
            result = self.runtime.step()
            # Acceptance aliases the reusable runtime buffer. Stage it on
            # device before the next cycle overwrites that buffer.
            token_history[index].copy_(result.acceptance.sampled_token_ids)
            count_history[index].copy_(self.runtime.state.num_sampled)
            cycles = index + 1
            progress = self._committed_progress()
            self._park_completed(progress)
            if all(
                progress[slot] - self._progress_baseline[slot]
                >= self.remaining[slot]
                for slot in range(cfg.batch_size)
            ):
                break
        else:
            raise RuntimeError("fixed cohort did not reach its output limits")

        tokens_cpu = token_history[:cycles].cpu()
        counts_cpu = count_history[:cycles].cpu()
        output: list[list[int]] = [[] for _ in range(cfg.batch_size)]
        staged = [0] * cfg.batch_size
        for cycle in range(cycles):
            for slot in range(cfg.batch_size):
                count = int(counts_cpu[cycle, slot])
                if count < 0 or count > width:
                    raise RuntimeError("acceptance count left the fixed contract")
                row = tokens_cpu[cycle, slot, :count].tolist()
                if any(token < 0 for token in row):
                    raise RuntimeError("accepted output contains padding")
                staged[slot] += count
                needed = self.remaining[slot] - len(output[slot])
                if needed > 0:
                    output[slot].extend(int(token) for token in row[:needed])

        acceptance_window_means = {}
        for start, end in ((0, 8), (8, 64), (64, 128),
                           (128, 192), (192, 256), (256, 512),
                           (512, 768), (768, 1024)):
            if start >= cycles:
                continue
            stop = min(end, cycles)
            acceptance_window_means[f"{start}-{stop - 1}"] = float(
                counts_cpu[start:stop].sum().item()
                / ((stop - start) * cfg.batch_size)
            )
        generated = [len(row) for row in output]
        if generated != self.remaining:
            raise RuntimeError(
                f"cohort output is incomplete: {generated} != {self.remaining}"
            )
        return FixedCohortOutput(
            token_ids=output,
            cycles=cycles,
            initial_output_counts=list(self.initial_output_counts),
            requested_output_counts=list(self.remaining),
            generated_output_counts=generated,
            staged_output_counts=staged,
            overshoot_tokens=sum(staged) - sum(generated),
            acceptance_window_means=acceptance_window_means,
        )
