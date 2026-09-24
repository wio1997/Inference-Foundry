"""Assemble the product runtime and sever the generic ModelRunner boundary."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from bootstrap.vllm_dspark_handoff import DSparkHandoffInputs, DirectDSparkHandoff
from bootstrap.vllm_target_handoff import DirectTargetHandoff, TargetHandoffInputs
from bootstrap.vllm_target_metadata_handoff import (
    TargetMetadataSource,
    bind_fixed_target_metadata,
)
from runtime.extreme_decode import ExtremeDecodeRuntime
from runtime.fixed_acceptance import FixedGreedyAcceptance
from runtime.fixed_decode import FixedDecodeConfig, FixedDecodeState
from runtime.target_adapter import FixedTargetAdapter
from runtime.target_metadata import FixedTargetMetadataUpdater


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
    target_metadata_updater: FixedTargetMetadataUpdater | None = None
    target_metadata_sources: tuple[TargetMetadataSource, ...] = ()
    target_tp_rank: int | None = None
    target_tp_size: int | None = None
    group_slot_audit_bindings: tuple = ()


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
    metadata_updater = inputs.target_metadata_updater
    if os.getenv("EXTREME_NATIVE_TARGET_METADATA") == "1":
        if metadata_updater is not None:
            raise ValueError("pass native metadata sources or updater, not both")
        if inputs.target_tp_rank is None or inputs.target_tp_size is None:
            raise ValueError("native target metadata needs TP rank and size")
        metadata_updater = bind_fixed_target_metadata(
            attn_metadata=target_handoff.attn_metadata,
            sources=inputs.target_metadata_sources,
            vllm_config=inputs.target.vllm_config,
            query_start_loc=state.target_query_start_loc,
            tp_rank=inputs.target_tp_rank,
            tp_size=inputs.target_tp_size,
        )
    kv_slot_audit = None
    if os.getenv("EXTREME_KV_SLOT_AUDIT_DIR"):
        from diagnostics.kv_slot_audit import KVSlotAudit
        kv_slot_audit = KVSlotAudit(
            inputs.group_slot_audit_bindings,
            os.environ["EXTREME_KV_SLOT_AUDIT_DIR"],
            int(inputs.target_tp_rank),
            int(os.getenv("EXTREME_KV_SLOT_AUDIT_CYCLES", "8")),
        )
    return ExtremeDecodeRuntime(
        config,
        state,
        FixedTargetAdapter(config, target_handoff.binding()),
        FixedGreedyAcceptance(config),
        DirectDSparkHandoff(config, inputs.dspark),
        target_metadata=metadata_updater,
        kv_slot_audit=kv_slot_audit,
    )
