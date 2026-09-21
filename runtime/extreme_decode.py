"""Product-owned continuous decode driver for DeepSeek Extreme.

This module is the runtime boundary.  It does not import vLLM and it does not
accept scheduler, request, batch, or metadata-builder objects.  A bootstrap
layer may lend it loaded operator callables and physical cache tensors once;
after that, this driver owns every decode-cycle transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

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
    ) -> None:
        self.config = config
        self.state = state
        self.target = target
        self.acceptance = acceptance
        self.proposer = proposer
        # Reuse the proven fixed-buffer preparation and state transition, not
        # its older bundled operator dispatch.
        self._state_machine = FixedDecodeRuntime(
            config,
            state,
            _UnreachableBundledOperators(),
        )

    @torch.inference_mode()
    def step(self) -> CycleResult:
        self._state_machine.prepare_target_inputs()
        target_output = self.target.execute(self.state)
        acceptance_output = self.acceptance.execute(self.state, target_output)
        self._state_machine.advance_state(acceptance_output)
        next_draft = self.proposer.execute(
            self.state,
            target_output,
            acceptance_output,
        )
        if tuple(next_draft.shape) != tuple(self.state.draft_tokens.shape):
            raise ValueError("DSpark draft tensor shape changed")
        self.state.draft_tokens.copy_(next_draft)
        self.state.cycle_index += 1
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
