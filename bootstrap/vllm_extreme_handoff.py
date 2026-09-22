"""Assemble the product runtime and sever the generic ModelRunner boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bootstrap.vllm_dspark_handoff import DSparkHandoffInputs, DirectDSparkHandoff
from bootstrap.vllm_target_handoff import DirectTargetHandoff, TargetHandoffInputs
from runtime.extreme_decode import ExtremeDecodeRuntime
from runtime.fixed_acceptance import FixedGreedyAcceptance
from runtime.fixed_decode import FixedDecodeConfig, FixedDecodeState
from runtime.target_adapter import FixedTargetAdapter


@dataclass(frozen=True)
class ExtremeHandoffInputs:
    target: TargetHandoffInputs
    dspark: DSparkHandoffInputs
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


def build_extreme_runtime(
    inputs: ExtremeHandoffInputs,
    config: FixedDecodeConfig | None = None,
) -> ExtremeDecodeRuntime:
    """Perform the one-time handoff; the result retains no ModelRunner."""

    config = config or FixedDecodeConfig()
    state = FixedDecodeState.bind(
        config,
        block_table=inputs.block_table,
        num_computed_tokens=inputs.num_computed_tokens,
        last_sampled_tokens=inputs.last_sampled_tokens,
        draft_tokens=inputs.draft_tokens,
        target_input_ids=inputs.target_input_ids,
        target_positions=inputs.target_positions,
        target_query_start_loc=inputs.target_query_start_loc,
        target_seq_lens=inputs.target_seq_lens,
        target_slot_mapping=inputs.target_slot_mapping,
        target_logits_indices=inputs.target_logits_indices,
    )
    target_handoff = DirectTargetHandoff(inputs.target)
    return ExtremeDecodeRuntime(
        config,
        state,
        FixedTargetAdapter(config, target_handoff.binding()),
        FixedGreedyAcceptance(config),
        DirectDSparkHandoff(config, inputs.dspark),
    )
