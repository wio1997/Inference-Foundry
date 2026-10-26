"""One-time transfer of fixed DSA-CP target metadata tensor destinations.

Only this bootstrap module inspects vLLM attention metadata and RoPE registry.
The returned updater keeps tensors, scalar constants and operator callables;
it does not retain a builder, ModelRunner, or metadata dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from runtime.target_metadata import (
    CPGroupBinding,
    FixedTargetMetadataUpdater,
    RotaryBinding,
    SparseMetadataOperators,
)
from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE


@dataclass(frozen=True)
class TargetMetadataSource:
    layer_name: str
    ratio: int
    block_size: int
    cu_seqlens_ori_kv: torch.Tensor
    cu_seqlens_cmp_kv: torch.Tensor
    seqused_q: torch.Tensor


def bind_fixed_target_metadata(
    *,
    attn_metadata: dict[str, Any],
    sources: tuple[TargetMetadataSource, ...],
    vllm_config: Any,
    query_start_loc: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> FixedTargetMetadataUpdater:
    groups: list[CPGroupBinding] = []
    seen: set[int] = set()
    scratch_source = None
    for source in sources:
        metadata = attn_metadata.get(source.layer_name)
        req = getattr(metadata, "req_metadata", None)
        cp = getattr(req, "cp_metadata", None)
        if cp is None or id(req) in seen:
            continue
        seen.add(id(req))
        ratio = source.ratio or 1
        if ratio not in (1, 4, 128):
            raise ValueError(f"unsupported fixed DSA ratio {ratio}")
        if scratch_source is None:
            scratch_source = source
        if ratio == 4 and req.qli_metadata is None:
            raise ValueError("c4 DSA-CP metadata has no QLI destination")
        if req.sas_metadata is None:
            raise ValueError(f"c{ratio} DSA-CP metadata has no SAS destination")
        groups.append(CPGroupBinding(
            ratio=ratio,
            seq_lens=req.seq_lens,
            input_positions=req.input_positions,
            start_pos=req.start_pos,
            local_query_start_loc=cp.local_query_start_loc,
            local_seq_lens=cp.local_seq_lens,
            sas_metadata=req.sas_metadata,
            qli_metadata=req.qli_metadata,
            swa_slot_mapping=req.slot_mapping if source.layer_name.endswith("swa_cache") else None,
            swa_block_table=req.block_table if source.layer_name.endswith("swa_cache") else None,
            swa_block_size=source.block_size if source.layer_name.endswith("swa_cache") else None,
        ))
    if not groups or scratch_source is None:
        raise RuntimeError("fixed DSA-CP bootstrap found no active groups")
    if {group.ratio for group in groups} != {1, 4, 128}:
        raise RuntimeError("fixed DSA-CP bootstrap requires c1, c4 and c128")

    rotary: list[RotaryBinding] = []
    first_cp = next(
        getattr(attn_metadata[source.layer_name].req_metadata, "cp_metadata")
        for source in sources
        if source.layer_name in attn_metadata
        and getattr(attn_metadata[source.layer_name], "req_metadata", None) is not None
    )
    for config_key, registered in _ROPE_STATE.registry_summary.items():
        if "default" not in registered:
            continue
        full = _ROPE_STATE.full_rope_cache.get(config_key)
        target = _ROPE_STATE.runtime_buffer.get(config_key, {}).get("default")
        if full is None or target is None:
            raise RuntimeError(f"fixed DSA RoPE buffer missing for {config_key}")
        local_cos = first_cp.local_cos._data[config_key]["default"][0]
        local_sin = first_cp.local_sin._data[config_key]["default"][1]
        per_rank = 96 // tp_size
        start = tp_rank * per_rank
        end = start + per_rank
        target_cos, target_sin = target
        cos_copy = (
            None if local_cos.data_ptr() == target_cos[start:end].data_ptr()
            else local_cos
        )
        sin_copy = (
            None if local_sin.data_ptr() == target_sin[start:end].data_ptr()
            else local_sin
        )
        rotary.append(RotaryBinding(
            full[0], full[1], target_cos, target_sin, cos_copy, sin_copy,
        ))
    if not rotary:
        raise RuntimeError("fixed DSA target RoPE configurations were not found")

    hf = vllm_config.model_config.hf_config
    ops = SparseMetadataOperators(
        sas=torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
        qli=torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata,
        cu_seqlens_ori_kv=scratch_source.cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=scratch_source.cu_seqlens_cmp_kv,
        seqused_q=scratch_source.seqused_q,
        device_name=str(query_start_loc.device),
        num_heads=hf.num_attention_heads,
        head_dim=vllm_config.model_config.get_head_size(),
        sliding_window=hf.sliding_window,
        index_topk=hf.index_topk,
        index_n_heads=hf.index_n_heads,
        index_head_dim=hf.index_head_dim,
    )
    return FixedTargetMetadataUpdater(
        tp_rank=tp_rank,
        tp_size=tp_size,
        query_start_loc=query_start_loc,
        groups=tuple(groups),
        rotary=tuple(rotary),
        operators=ops,
    )
