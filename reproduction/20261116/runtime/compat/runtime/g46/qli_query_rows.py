"""Opt-in QLI-only row partition primitives; no installation or process groups.

All projections/quantization and the full KV/compressor remain caller-owned.
The caller must restore full row order before attention/candidate consumers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRows:
    total_rows: int
    global_end: int
    rank: int
    world: int
    ratio: int
    start: int
    end: int
    causal_end: int

    @property
    def rows(self):
        return self.end - self.start

    @property
    def compressed_end(self):
        return self.causal_end // self.ratio

    @property
    def residual(self):
        return self.causal_end % self.ratio


def plan_query_rows(total_rows, global_end, rank, world=4, ratio=1):
    """Contiguous ceil partitions of one request's consecutive absolute rows."""
    if (total_rows <= 0 or global_end < total_rows or world <= 0
            or not 0 <= rank < world or ratio not in (1, 2)):
        raise ValueError('Invalid single-request QLI row geometry')
    chunk = (total_rows + world - 1) // world
    start = min(total_rows, rank * chunk)
    end = min(total_rows, start + chunk)
    return QueryRows(total_rows, global_end, rank, world, ratio,
                     start, end, global_end - total_rows + end)


def build_local_qli_metadata(bindings, plan, device, *, causal_end=None,
                             heads=32, head_dim=128, topk=512):
    """Independent local metadata; never mutate full attention metadata.

    causal_end override exists solely for a deliberately incorrect probe control.
    """
    import torch
    if plan.rows <= 0:
        raise ValueError('Empty QLI partitions must bypass kernel execution')
    end = plan.causal_end if causal_end is None else causal_end
    cu = torch.tensor([0, plan.rows], dtype=torch.int32, device=device)
    seq = torch.tensor([end // plan.ratio], dtype=torch.int32, device=device)
    residual = (torch.tensor([end % plan.ratio], dtype=torch.int32, device=device)
                if plan.ratio == 2 else None)
    metadata = bindings.qli_metadata(
        heads, 1, head_dim, topk, 2, cu_seqlens_q=cu, seqused_k=seq,
        cmp_residual_k=residual, batch_size=1, max_seqlen_q=plan.rows,
        max_seqlen_k=max(1, end // plan.ratio), layout_q='TND',
        layout_k='PA_BBND', mask_mode=3, cmp_ratio=plan.ratio)
    return dict(cu=cu, seq=seq, residual=residual, metadata=metadata)


def call_local_qli(bindings, plan, q, q_scale, weights, key, key_scale,
                   block_table, local_metadata, *, mode, candidates=None,
                   topk=512, candidate_blocks=2048, candidate_block_size=8):
    """Slice already fully projected/quantized rows and call original QLI.

    q/q_scale/weights and candidates are FULL-row tensors. K stays unsliced.
    Return raw operator outputs; finalize/gather are caller responsibilities.
    """
    if mode not in (1, 2, 3):
        raise ValueError('Unknown candidate mode')
    if q.shape[0] != plan.total_rows or q_scale.shape[0] != plan.total_rows or weights.shape[0] != plan.total_rows:
        raise ValueError('Expected full-row query operands before partition')
    if mode == 2 and (candidates is None or candidates.shape[0] != plan.total_rows):
        raise ValueError('Consumer needs full-row baseline candidates')
    sl = slice(plan.start, plan.end)
    selected, values, published = bindings.qli(
        q[sl].contiguous(), key, weights[sl].contiguous(),
        q_scale[sl].contiguous(), key_scale.squeeze(-1), topk, 2,
        block_table=block_table, metadata=local_metadata['metadata'],
        candidate_topk_index=candidates[sl].contiguous() if mode == 2 else None,
        candidate_mode=mode, candidate_topk_blocks=candidate_blocks,
        candidate_block_size=candidate_block_size,
        cu_seqlens_q=local_metadata['cu'], seqused_k=local_metadata['seq'],
        cmp_residual_k=local_metadata['residual'], max_seqlen_q=plan.rows,
        layout_q='TND', layout_k='PA_BBND', mask_mode=3, cmp_ratio=plan.ratio)
    return selected, published
