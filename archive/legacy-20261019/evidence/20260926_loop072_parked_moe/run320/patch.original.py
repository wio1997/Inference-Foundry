"""Assemble the product runtime and sever the generic ModelRunner boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

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
    if os.getenv("EXTREME_CACHE_MANIFEST_DIR"):
        if len(inputs.group_slot_audit_bindings) == 0:
            raise RuntimeError("cache manifest requires all group slot bindings")
        group_rows = []
        mapping_to_gid = {}
        table_to_gid = {}
        for gid, (table, mapping, size) in enumerate(inputs.group_slot_audit_bindings):
            group_rows.append({"gid": gid, "block_size": int(size),
                               "table_shape": list(table.shape),
                               "mapping_shape": list(mapping.shape),
                               "table_ptr": int(table.data_ptr()),
                               "mapping_ptr": int(mapping.data_ptr())})
            mapping_to_gid[int(mapping.data_ptr())] = gid
            table_to_gid[int(table.data_ptr())] = gid
        cache_rows = []
        layer_aliases = {}
        static_context = inputs.target.vllm_config.compilation_config.static_forward_context
        def visit_cache(value, layer_name):
            if torch.is_tensor(value):
                layer_aliases.setdefault(int(value.data_ptr()), set()).add(layer_name)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit_cache(child, layer_name)
        for layer_name, layer in static_context.items():
            value = getattr(layer, "kv_cache", None)
            if value is not None:
                visit_cache(value, layer_name)
        for cache in target_handoff.assets.caches:
            spec = target_handoff.assets._cache_slot_specs.get(cache.name)
            aliases = []
            for layer_name in sorted(layer_aliases.get(int(cache.data_ptr), ())):
                metadata = target_handoff.attn_metadata.get(layer_name)
                req = getattr(metadata, "req_metadata", None)
                table = getattr(req, "block_table", None)
                aliases.append({
                    "layer": layer_name,
                    "table_group": None if table is None else table_to_gid.get(int(table.data_ptr())),
                    "table_ptr": None if table is None else int(table.data_ptr()),
                })
            cache_rows.append({"name": cache.name, "shape": list(cache.shape),
                               "stride": list(cache.stride),
                               "dtype": str(cache.tensor.dtype),
                               "group": None if spec is None else mapping_to_gid.get(int(spec[0].data_ptr())),
                               "slot_block_size": None if spec is None else int(spec[1]),
                               "has_slot_spec": spec is not None,
                               "storage_ptr": int(cache.tensor.untyped_storage().data_ptr()),
                               "storage_offset": int(cache.tensor.storage_offset()),
                               "layer_aliases": aliases})
        source_rows = []
        for source in inputs.target_metadata_sources:
            metadata = target_handoff.attn_metadata.get(source.layer_name)
            req = getattr(metadata, "req_metadata", None)
            table = getattr(req, "block_table", None)
            slots = getattr(req, "slot_mapping", None)
            source_rows.append({"layer": source.layer_name,
                                "ratio": int(source.ratio),
                                "block_size": int(source.block_size),
                                "metadata_type": None if metadata is None else type(metadata).__name__,
                                "req_metadata_type": None if req is None else type(req).__name__,
                                "table_shape": None if table is None else list(table.shape),
                                "table_group": None if table is None else table_to_gid.get(int(table.data_ptr())),
                                "slot_shape": None if slots is None else list(slots.shape),
                                "slot_ptr": None if slots is None else int(slots.data_ptr())})
        draft_rows = [{"gid": int(gid), "block_size": int(size),
                       "table_group": table_to_gid.get(int(table.data_ptr())),
                       "mapping_group": mapping_to_gid.get(int(mapping.data_ptr()))}
                      for gid, table, mapping, size in inputs.dspark.group_slot_bindings]
        payload = {"rank": int(inputs.target_tp_rank), "groups": group_rows,
                   "target_caches": cache_rows, "target_sources": source_rows,
                   "draft_groups": draft_rows}
        output_dir = Path(os.environ["EXTREME_CACHE_MANIFEST_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"rank{inputs.target_tp_rank}.json").write_text(
            json.dumps(payload, indent=2) + "\n")
    target_page_audit = None
    if os.getenv("EXTREME_CROSS_LAYER_LIVENESS_DIR"):
        from diagnostics.cross_layer_liveness_census import CrossLayerLivenessCensus
        target_page_audit = CrossLayerLivenessCensus(
            static_context=inputs.target.vllm_config.compilation_config.static_forward_context,
            attn_metadata=target_handoff.attn_metadata,
            target_model=inputs.target.model,
            proposer=inputs.dspark.proposer,
            metadata_sources=inputs.target_metadata_sources,
            output_dir=os.environ["EXTREME_CROSS_LAYER_LIVENESS_DIR"],
            rank=int(inputs.target_tp_rank),
        )
    if os.getenv("EXTREME_STORAGE_LIVENESS_DIR"):
        if target_page_audit is not None:
            raise ValueError("only one liveness diagnostic may be enabled")
        from diagnostics.storage_liveness_census import StorageLivenessCensus
        target_page_audit = StorageLivenessCensus(
            static_context=inputs.target.vllm_config.compilation_config.static_forward_context,
            assets=target_handoff.assets,
            attn_metadata=target_handoff.attn_metadata,
            proposer=inputs.dspark.proposer,
            group_bindings=inputs.group_slot_audit_bindings,
            output_dir=os.environ["EXTREME_STORAGE_LIVENESS_DIR"],
            rank=int(inputs.target_tp_rank),
        )
    if os.getenv("EXTREME_TARGET_PAGE_AUDIT_DIR"):
        if target_page_audit is not None:
            raise ValueError("storage liveness and page audit cannot be enabled together")
        from diagnostics.target_page_audit import TargetPageAudit
        target_page_audit = TargetPageAudit(
            assets=target_handoff.assets,
            attn_metadata=target_handoff.attn_metadata,
            static_context=inputs.target.vllm_config.compilation_config.static_forward_context,
            group_bindings=inputs.group_slot_audit_bindings,
            metadata_sources=inputs.target_metadata_sources,
            output_dir=os.environ["EXTREME_TARGET_PAGE_AUDIT_DIR"],
            rank=int(inputs.target_tp_rank),
            limit=int(os.getenv("EXTREME_TARGET_PAGE_AUDIT_CYCLES", "2")),
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
        target_page_audit=target_page_audit,
        protected_kv_tensors=(
            tuple(cache.tensor for cache in target_handoff.assets.caches)
            if os.getenv("EXTREME_SCHEDULE_NEXT_TARGET_METADATA", "off") != "off"
            else ()
        ),
    )
