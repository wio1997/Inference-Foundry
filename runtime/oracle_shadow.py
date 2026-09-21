"""Bridge used only to bootstrap and validate the standalone state machine.

The oracle still executes the model in this phase.  This shadow consumes its
small boundary tensors and predicts the *next* target input block using the
standalone runtime state.  A match over consecutive cycles establishes the
state-transition contract required before the target operator is moved behind
the dedicated runtime entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fixed_decode import (
    AcceptanceOutput,
    FixedDecodeConfig,
    FixedDecodeRuntime,
    FixedDecodeState,
)


class _NoOperators:
    """The shadow uses state preparation/advance only, never model callbacks."""


@dataclass
class ShadowComparison:
    cycle: int
    input_ids_equal: bool
    positions_equal: bool
    query_start_equal: bool
    seq_lens_equal: bool | None
    slot_mapping_equal: bool | None

    @property
    def exact(self) -> bool:
        fields = (
            self.input_ids_equal,
            self.positions_equal,
            self.query_start_equal,
            self.seq_lens_equal,
            self.slot_mapping_equal,
        )
        return all(value is not False for value in fields)


class FixedDecodeOracleShadow:
    def __init__(self, config: FixedDecodeConfig) -> None:
        self.config = config
        self.runtime: FixedDecodeRuntime | None = None
        self.comparisons: list[ShadowComparison] = []

    def observe_target_inputs(
        self,
        *,
        block_table: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> ShadowComparison | None:
        cfg = self.config
        width = cfg.target_tokens_per_request
        total = cfg.target_token_count
        ids = input_ids[:total].view(cfg.batch_size, width)
        pos = positions[:total].view(cfg.batch_size, width)

        if self.runtime is None:
            state = FixedDecodeState.allocate(
                cfg,
                block_table=block_table,
                initial_num_computed_tokens=pos[:, 0].to(torch.int32).clone(),
                initial_last_sampled_tokens=ids[:, 0].to(torch.int64).clone(),
                initial_draft_tokens=ids[:, 1:].to(torch.int64).clone(),
            )
            self.runtime = FixedDecodeRuntime(cfg, state, _NoOperators())  # type: ignore[arg-type]
            return None

        state = self.runtime.state
        result = ShadowComparison(
            cycle=state.cycle_index,
            input_ids_equal=bool(torch.equal(state.target_input_ids, input_ids[:total])),
            positions_equal=bool(torch.equal(state.target_positions, positions[:total])),
            query_start_equal=bool(
                torch.equal(
                    state.target_query_start_loc,
                    query_start_loc[: cfg.batch_size + 1],
                )
            ),
            seq_lens_equal=(
                None
                if seq_lens is None
                else bool(torch.equal(state.target_seq_lens, seq_lens[: cfg.batch_size]))
            ),
            slot_mapping_equal=(
                None
                if slot_mapping is None
                else bool(
                    torch.equal(
                        state.target_slot_mapping,
                        slot_mapping[:total].to(state.target_slot_mapping.dtype),
                    )
                )
            ),
        )
        self.comparisons.append(result)
        return result

    def observe_cycle_outputs(
        self,
        sampled_token_ids: torch.Tensor,
        next_draft_tokens: torch.Tensor,
    ) -> None:
        if self.runtime is None:
            raise RuntimeError("target inputs must be observed before cycle outputs")
        cfg = self.config
        sampled = sampled_token_ids[: cfg.batch_size, : cfg.target_tokens_per_request]
        counts = sampled.ne(-1).sum(dim=1).to(torch.int32)
        acceptance = AcceptanceOutput(sampled, counts)
        self.runtime.advance_state(acceptance)
        self.runtime.state.draft_tokens.copy_(
            next_draft_tokens[: cfg.batch_size, : cfg.speculative_tokens]
        )
        self.runtime.state.cycle_index += 1
        self.runtime.prepare_target_inputs()
