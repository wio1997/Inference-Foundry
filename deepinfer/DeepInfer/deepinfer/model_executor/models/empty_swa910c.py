"""Adapt a C1-only YOCO query to the existing BF16 sparse MLA ABI."""

import torch


def compressed_only(query, batch, sink, cache, table, selected):
    """C1 and ordinary BF16 pages have the same logical address space."""
    from deepinfer.npu_ops.dsa910c import load_ops

    if (query.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16
            or query.ndim != 3 or query.shape[-1] != 512 or cache.shape[1:] != (128, 1, 512)
            or query.shape[0] != sum(batch.query_lengths)
            or selected.shape != (query.shape[0], 512) or selected.dtype != torch.int32):
        raise ValueError("invalid C1-only YOCO geometry")
    load_ops()
    lengths = (selected >= 0).sum(dim=-1, dtype=torch.int32, keepdim=True).contiguous()
    common = dict(cu_seqlens_q=batch.cu_query, seqused_ori_kv=batch.seq_used,
                  seqused_cmp_kv=None, cmp_residual_kv=None, cmp_ratio=0,
                  ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=127, ori_win_right=0,
                  layout_q="TND", layout_kv="PA_BBND")
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        query.shape[1], 1, 512, batch_size=len(batch.query_lengths),
        max_seqlen_q=max(batch.query_lengths), max_seqlen_ori_kv=max(batch.seq_lengths),
        max_seqlen_cmp_kv=0, ori_topk=512, ori_topk_length=lengths, cmp_topk=0,
        has_ori_kv=True, has_cmp_kv=False, **common)
    out, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
        query, ori_kv=cache, cmp_kv=None, ori_sparse_indices=selected[:, None, :],
        ori_topk_length=lengths, cmp_sparse_indices=None, ori_block_table=table,
        cmp_block_table=None, sinks=sink, metadata=metadata, softmax_scale=512**-.5,
        topk_value_mode=1, return_softmax_lse=False, **common)
    return out
