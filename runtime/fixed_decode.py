"""Fixed-buffer decode cycle for DeepSeek Extreme P0.

This module intentionally does not depend on SchedulerOutput, request objects,
InputBatch, ModelRunner, or metadata builders.  Weight loading and the first
set of operator adapters may come from the oracle process, but the cycle state
and ordering belong to this runtime.

The first product envelope is deliberately narrow: greedy DSpark7 decoding at
c12 on DP1 x TP8.  Dynamic batching and compatibility branches are not part of
this ABI.  Request completion/refill will be represented later by a fixed slot
mask rather than by changing tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class FixedDecodeConfig:
    batch_size: int = 12
    speculative_tokens: int = 7
    block_size: int = 32
    max_model_len: int = 1_048_576

    @property
    def target_tokens_per_request(self) -> int:
        return self.speculative_tokens + 1

    @property
    def target_token_count(self) -> int:
        return self.batch_size * self.target_tokens_per_request


@dataclass
class TargetOutput:
    """Target outputs consumed by verification and the DSpark proposer."""

    logits: torch.Tensor
    hidden_states: torch.Tensor
    aux_hidden_states: tuple[torch.Tensor, ...]


@dataclass
class AcceptanceOutput:
    """Fixed-width accepted output; unused columns contain -1."""

    sampled_token_ids: torch.Tensor
    num_sampled: torch.Tensor


@dataclass
class FixedDecodeState:
    """All mutable state required by the fixed decode loop.

    KV and recurrent-state storage are owned by the operator adapters because
    their physical layouts are backend-specific.  Their mutations are still
    part of cycle parity and are surfaced through adapter fingerprints.
    """

    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    last_sampled_tokens: torch.Tensor
    draft_tokens: torch.Tensor

    target_input_ids: torch.Tensor
    target_positions: torch.Tensor
    target_query_start_loc: torch.Tensor
    target_seq_lens: torch.Tensor
    target_slot_mapping: torch.Tensor
    target_logits_indices: torch.Tensor

    accepted_tokens: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    emitted_token_count: torch.Tensor

    cycle_index: int = 0

    @classmethod
    def bind(
        cls,
        config: FixedDecodeConfig,
        *,
        block_table: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_query_start_loc: torch.Tensor,
        target_seq_lens: torch.Tensor,
        target_slot_mapping: torch.Tensor,
        target_logits_indices: torch.Tensor,
        accepted_tokens: torch.Tensor | None = None,
        num_sampled: torch.Tensor | None = None,
        num_rejected: torch.Tensor | None = None,
        emitted_token_count: torch.Tensor | None = None,
    ) -> "FixedDecodeState":
        """Take over bootstrap buffers without copying or retaining an owner."""

        batch = config.batch_size
        width = config.target_tokens_per_request
        total = config.target_token_count
        required = {
            "block_table": (block_table, batch),
            "num_computed_tokens": (num_computed_tokens, batch),
            "last_sampled_tokens": (last_sampled_tokens, batch),
            "draft_tokens": (draft_tokens, batch * config.speculative_tokens),
            "target_input_ids": (target_input_ids, total),
            "target_positions": (target_positions, total),
            "target_query_start_loc": (target_query_start_loc, batch + 1),
            "target_seq_lens": (target_seq_lens, batch),
            "target_slot_mapping": (target_slot_mapping, total),
            "target_logits_indices": (target_logits_indices, total),
        }
        for name, (tensor, minimum) in required.items():
            if tensor.numel() < minimum:
                raise ValueError(f"{name} bootstrap buffer is too small")

        device = target_input_ids.device
        if accepted_tokens is None:
            accepted_tokens = torch.full(
                (batch, width), -1, dtype=torch.int64, device=device
            )
        if num_sampled is None:
            num_sampled = torch.empty(batch, dtype=torch.int32, device=device)
        if num_rejected is None:
            num_rejected = torch.empty(batch, dtype=torch.int32, device=device)
        if emitted_token_count is None:
            emitted_token_count = torch.zeros(
                batch, dtype=torch.int32, device=device
            )
        return cls(
            block_table=block_table[:batch],
            num_computed_tokens=num_computed_tokens[:batch],
            last_sampled_tokens=last_sampled_tokens[:batch],
            draft_tokens=draft_tokens[:batch, : config.speculative_tokens],
            target_input_ids=target_input_ids[:total],
            target_positions=target_positions[:total],
            target_query_start_loc=target_query_start_loc[: batch + 1],
            target_seq_lens=target_seq_lens[:batch],
            target_slot_mapping=target_slot_mapping[:total],
            target_logits_indices=target_logits_indices[:total],
            accepted_tokens=accepted_tokens,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            emitted_token_count=emitted_token_count,
        )

    @classmethod
    def allocate(
        cls,
        config: FixedDecodeConfig,
        block_table: torch.Tensor,
        initial_num_computed_tokens: torch.Tensor,
        initial_last_sampled_tokens: torch.Tensor,
        initial_draft_tokens: torch.Tensor,
    ) -> "FixedDecodeState":
        device = block_table.device
        batch = config.batch_size
        width = config.target_tokens_per_request
        total = config.target_token_count

        if tuple(block_table.shape[:1]) != (batch,):
            raise ValueError(f"block_table must have {batch} rows")
        if tuple(initial_num_computed_tokens.shape) != (batch,):
            raise ValueError("initial_num_computed_tokens shape mismatch")
        if tuple(initial_last_sampled_tokens.shape) != (batch,):
            raise ValueError("initial_last_sampled_tokens shape mismatch")
        if tuple(initial_draft_tokens.shape) != (batch, config.speculative_tokens):
            raise ValueError("initial_draft_tokens shape mismatch")

        query_start = torch.arange(
            0,
            total + 1,
            width,
            dtype=torch.int32,
            device=device,
        )
        return cls(
            block_table=block_table,
            num_computed_tokens=initial_num_computed_tokens,
            last_sampled_tokens=initial_last_sampled_tokens,
            draft_tokens=initial_draft_tokens,
            target_input_ids=torch.empty(total, dtype=torch.int32, device=device),
            target_positions=torch.empty(total, dtype=torch.int64, device=device),
            target_query_start_loc=query_start,
            target_seq_lens=torch.empty(batch, dtype=torch.int32, device=device),
            target_slot_mapping=torch.empty(total, dtype=torch.int32, device=device),
            target_logits_indices=torch.arange(total, dtype=torch.int64, device=device),
            accepted_tokens=torch.full(
                (batch, width), -1, dtype=torch.int64, device=device
            ),
            num_sampled=torch.empty(batch, dtype=torch.int32, device=device),
            num_rejected=torch.empty(batch, dtype=torch.int32, device=device),
            emitted_token_count=torch.zeros(batch, dtype=torch.int32, device=device),
        )


class DecodeOperators(Protocol):
    """Backend adapter for real DeepSeek target/DSpark operators and caches."""

    def target_forward(self, state: FixedDecodeState) -> TargetOutput: ...

    def verify(
        self, state: FixedDecodeState, target: TargetOutput
    ) -> AcceptanceOutput: ...

    def propose(
        self,
        state: FixedDecodeState,
        target: TargetOutput,
        acceptance: AcceptanceOutput,
    ) -> torch.Tensor: ...

    def state_fingerprint(self, state: FixedDecodeState) -> dict[str, torch.Tensor]: ...


class FixedDecodeRuntime:
    """Owns the fixed decode hot path and its preallocated device buffers."""

    def __init__(
        self,
        config: FixedDecodeConfig,
        state: FixedDecodeState,
        operators: DecodeOperators,
    ) -> None:
        self.config = config
        self.state = state
        self.operators = operators
        self._request_index = torch.arange(
            config.batch_size, dtype=torch.int64, device=state.block_table.device
        ).unsqueeze(1)
        self._position_offsets = torch.arange(
            config.target_tokens_per_request,
            dtype=torch.int64,
            device=state.block_table.device,
        ).unsqueeze(0)

    def prepare_target_inputs(self) -> None:
        """Materialize the next fixed 12x8 target verification block on device."""

        cfg = self.config
        state = self.state
        width = cfg.target_tokens_per_request

        ids = state.target_input_ids.view(cfg.batch_size, width)
        ids[:, 0].copy_(state.last_sampled_tokens.to(torch.int32))
        ids[:, 1:].copy_(state.draft_tokens.to(torch.int32))

        positions = state.target_positions.view(cfg.batch_size, width)
        torch.add(
            state.num_computed_tokens.to(torch.int64).unsqueeze(1),
            self._position_offsets,
            out=positions,
        )
        state.target_seq_lens.copy_(
            state.num_computed_tokens + cfg.target_tokens_per_request
        )
        logical_blocks = torch.div(
            positions, cfg.block_size, rounding_mode="floor"
        ).to(torch.int64)
        physical_blocks = state.block_table[self._request_index, logical_blocks]
        slots = physical_blocks.to(torch.int64) * cfg.block_size
        slots.add_(positions.remainder(cfg.block_size))
        state.target_slot_mapping.copy_(slots.flatten().to(torch.int32))

    def advance_state(self, acceptance: AcceptanceOutput) -> None:
        """Commit acceptance entirely in fixed device tensors."""

        cfg = self.config
        state = self.state
        sampled = acceptance.sampled_token_ids
        if tuple(sampled.shape) != (
            cfg.batch_size,
            cfg.target_tokens_per_request,
        ):
            raise ValueError("accepted token tensor shape mismatch")

        state.accepted_tokens.copy_(sampled)
        state.num_sampled.copy_(acceptance.num_sampled.to(torch.int32))
        state.num_rejected.copy_(
            cfg.target_tokens_per_request - state.num_sampled
        )

        last_index = (state.num_sampled - 1).clamp_min(0).to(torch.int64)
        last = torch.gather(sampled, 1, last_index.unsqueeze(1)).squeeze(1)
        has_output = state.num_sampled > 0
        state.last_sampled_tokens.copy_(
            torch.where(has_output, last, state.last_sampled_tokens).to(
                state.last_sampled_tokens.dtype
            )
        )
        state.num_computed_tokens.add_(state.num_sampled)
        state.emitted_token_count.add_(state.num_sampled)

    @torch.inference_mode()
    def run_cycle(self) -> AcceptanceOutput:
        self.prepare_target_inputs()
        target = self.operators.target_forward(self.state)
        acceptance = self.operators.verify(self.state, target)
        self.advance_state(acceptance)
        next_draft = self.operators.propose(
            self.state, target, acceptance
        )
        if tuple(next_draft.shape) != tuple(self.state.draft_tokens.shape):
            raise ValueError("DSpark draft tensor shape mismatch")
        self.state.draft_tokens.copy_(next_draft)
        self.state.cycle_index += 1
        return acceptance

    @torch.inference_mode()
    def run(self, cycles: int) -> list[AcceptanceOutput]:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        return [self.run_cycle() for _ in range(cycles)]
