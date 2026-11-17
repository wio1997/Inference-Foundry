# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project; Copyright SGLang Team.
# SGLang native V4.1 state ownership plus installed vLLM-Ascend A3 storage ABI.
# Component candidate only: not a complete attention backend or registry patch.
"""Reuse native pair/history logic; use verified A3 BF16/INT8 source storage.

Caller owns metadata and graph-lifetime buffers. This module does not install
itself, create a service/process group, or instantiate a vLLM runtime.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class A3SourceMetadata:
    # All coordinates follow the actual installed A3 QLI caller.
    query_start_loc: torch.Tensor  # [bs+1] int32 cumulative query lengths
    cache_seq_lens: torch.Tensor  # [bs] completed compressed rows
    cmp_residual: Optional[torch.Tensor]
    block_table: torch.Tensor  # [bs,pages] same physical full-page IDs
    qli_metadata: torch.Tensor
    max_query_len: int


def rotate_tail_inplace(tensor, layer, positions):
    """Existing native NPU RoPE kernel; caches must already be primed at init."""
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE
    rope = Dsv4NpuRoPE.for_freqs(layer.freqs_cis, getattr(layer, 'rotary_emb', None))
    cos, sin = rope.get_cos_sin(positions, tensor.dtype, view_4d=True,
                               allow_build=False, cache_dtype=tensor.dtype)
    Dsv4NpuRoPE.apply_rotary_mul_inplace(tensor, None, cos, sin,
                                       qk_nope_dim=tensor.shape[-1]-layer.rope_head_dim)
    return tensor


def write_a3_source_group(pool, layer, pooled, slots, group_pos):
    """Native post-pooling norm; publish index K BEFORE rotating long KV.

    slots are compressed physical locations. Cache adapter owns skipping slot0
    and negative locations. This deliberately does not call generic fake-FP4.
    """
    import torch_npu
    latent = layer.compressor.finish(pooled)
    if layer.indexer is not None and layer.indexer.owns_k:
        indexer = layer.indexer
        key = indexer.k_norm(indexer.forward_wk(latent)).view(-1, 1, indexer.index_head_dim)
        # Pass the attention layer: both use the same RoPE tail and table.
        rotate_tail_inplace(key, layer, group_pos)
        codes, scales = torch_npu.npu_dynamic_quant(key.squeeze(1), dst_type=torch.int8)
        pool.set_index_kv_npu(layer.layer_id, slots, codes,
                              scales.unsqueeze(-1).to(torch.float16))
    long_kv = latent.view(-1, 1, layer.head_dim)
    rotate_tail_inplace(long_kv, layer, group_pos)
    pool.set_extra_key_buffer_fused(layer_id=layer.layer_id,loc=slots,cache_k=long_kv)



def project_a3_index_query_weights(indexer, layer, hidden_states, q_lora, positions):
    """Preserve A3 FP32 weight scaling, before the QLI FP16 ABI cast."""
    query, _ = indexer.wq_b(q_lora)
    query = query.view(-1, indexer.n_heads, indexer.index_head_dim)
    rotate_tail_inplace(query, layer, positions)
    raw_weights, _ = indexer.weights_proj(hidden_states)
    weights = raw_weights.float() * indexer.head_weight_scale
    return query, weights


def run_a3_qli(query, weights, positions, source_cache, source_metadata,
               *, compress_ratio, index_topk, is_candidate_source,
               uses_candidate_filter, candidate_topk_blocks, candidate_block_size,
               candidates, quantize_query:Callable, finalize_indices:Callable,
               indices_output=None, lengths_output=None, qli_op=None):
    """Exact installed A3 QLI-v3 argument ABI, without vLLM state/context.

    quantize_query and finalize_indices must be copied from the pinned installed
    reference or independently verified equivalents. No fallback to different
    precision/candidate semantics is selected here.
    """
    if compress_ratio not in (1,2):
        raise ValueError('A3 V4.1 QLI component accepts ratios1/2 only')
    key,key_scale = source_cache
    if key.dtype!=torch.int8 or key_scale.dtype!=torch.float16:
        raise ValueError('A3 QLI requires INT8 K and FP16 scales')
    if source_metadata.qli_metadata is None:
        raise ValueError('QLI metadata must be built outside this component')
    q,qs = quantize_query(query)
    mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
    op = torch.ops._C_ascend.npu_quant_lightning_indexer_v3 if qli_op is None else qli_op
    selected,_,candidate_out = op(
        q,key,weights.to(torch.float16),qs,key_scale.squeeze(-1),index_topk,2,
        block_table=source_metadata.block_table,metadata=source_metadata.qli_metadata,
        candidate_topk_index=candidates if uses_candidate_filter else None,
        candidate_mode=mode,candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        cu_seqlens_q=source_metadata.query_start_loc,
        seqused_k=source_metadata.cache_seq_lens,
        cmp_residual_k=source_metadata.cmp_residual,
        max_seqlen_q=source_metadata.max_query_len,
        layout_q='TND',layout_k='PA_BBND',mask_mode=3,cmp_ratio=compress_ratio)
    selected = finalize_indices(selected.squeeze(1),positions,compress_ratio,
                                 indices_output=indices_output,lengths_output=lengths_output)
    return selected,candidate_out if is_candidate_source else candidates


def native_pair_source_mixin():
    """Construct a component mixin without copying/reimplementing native state.

    Integration supplies forward_metadata.core_metadata.{raw_out_loc,c1_out_loc,
    c2_out_loc}, token_to_kv_pool, and the ordinary native layer/compressor.
    Metadata allocation, request maps, graph replay and attention remain outside
    this component; do not register it as a complete backend.
    """
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    class A3NativePairSourceMixin:
        _low_ratio_pair_partners = DeepseekV4AttnBackend._low_ratio_pair_partners
        _low_ratio_compress_torch = DeepseekV4AttnBackend._low_ratio_compress_torch
        def _low_ratio_write_group(self,layer,pooled,slots,group_pos,*,fuse_index_store=False):
            return write_a3_source_group(self.token_to_kv_pool,layer,pooled,slots,group_pos)
    return A3NativePairSourceMixin
