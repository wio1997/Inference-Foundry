"""Candidate that loads each shared-KV tile once per pair of query heads."""
import torch
import triton
import triton.language as tl

from lowratio_sparse_attention import BANK_ROWS


@triton.jit
def _update_head(q, kv, valid, maximum, denominator, accumulator, scale: tl.constexpr):
    score = tl.sum(kv * q[None, :], 1) * scale
    score = tl.where(valid, score, float('-inf'))
    next_maximum = tl.maximum(maximum, tl.max(score, 0))
    alpha = tl.exp(tl.where(next_maximum == float('-inf'), 0.0, maximum - next_maximum))
    probability = tl.exp(tl.where(next_maximum == float('-inf'), float('-inf'), score - next_maximum))
    accumulator = accumulator * alpha + tl.sum(probability[:, None] * kv, 0)
    denominator = denominator * alpha + tl.sum(probability, 0)
    return next_maximum, denominator, accumulator


@triton.jit
def _sparse_shared_kv_two_heads(
    Q, ORI_PTRS, CMP_PTRS, OI, CI, SINK, OUT,
    Q_ROW: tl.constexpr, ORI_ROW: tl.constexpr, CMP_ROW: tl.constexpr,
    OI_ROW: tl.constexpr, CI_ROW: tl.constexpr, OUT_ROW: tl.constexpr,
    OBANKS: tl.constexpr, CBANKS: tl.constexpr, BANK: tl.constexpr,
    DIM: tl.constexpr, SWA: tl.constexpr, TOPK: tl.constexpr,
    SCALE: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = 2 * tl.program_id(1).to(tl.int64)
    d = tl.arange(0, BD)
    n = tl.arange(0, BN)
    q0 = tl.load(Q + row * Q_ROW + head * DIM + d, d < DIM, other=0).to(tl.float32)
    q1 = tl.load(Q + row * Q_ROW + (head + 1) * DIM + d, d < DIM, other=0).to(tl.float32)
    maximum0 = tl.load(SINK + head).to(tl.float32)
    maximum1 = tl.load(SINK + head + 1).to(tl.float32)
    denominator0 = tl.where(maximum0 == float('-inf'), 0.0, 1.0)
    denominator1 = tl.where(maximum1 == float('-inf'), 0.0, 1.0)
    accumulator0 = tl.full((BD,), 0, tl.float32)
    accumulator1 = tl.full((BD,), 0, tl.float32)

    for block in range(tl.cdiv(SWA + TOPK, BN)):
        col = block * BN + n
        oi = tl.load(OI + row * OI_ROW + col, col < SWA, other=-1)
        ci = tl.load(CI + row * CI_ROW + col - SWA, (col >= SWA) & (col < SWA + TOPK), other=-1)
        ko = tl.full((BN, BD), 0, tl.float32)
        kc = tl.full((BN, BD), 0, tl.float32)
        for bank in range(OBANKS):
            in_bank = (col < SWA) & (oi >= bank * BANK) & (oi < (bank + 1) * BANK)
            if tl.sum(in_bank.to(tl.int32), 0) > 0:
                base = tl.load(ORI_PTRS + bank).to(tl.pointer_type(tl.bfloat16))
                local = tl.where(in_bank, oi - bank * BANK, 0)
                part = tl.load(base + local[:, None] * ORI_ROW + d[None, :],
                               in_bank[:, None] & (d[None, :] < DIM), other=0)
                ko += part.to(tl.float32)
        for bank in range(CBANKS):
            in_bank = (col >= SWA) & (col < SWA + TOPK) & (ci >= bank * BANK) & (ci < (bank + 1) * BANK)
            if tl.sum(in_bank.to(tl.int32), 0) > 0:
                base = tl.load(CMP_PTRS + bank).to(tl.pointer_type(tl.bfloat16))
                local = tl.where(in_bank, ci - bank * BANK, 0)
                part = tl.load(base + local[:, None] * CMP_ROW + d[None, :],
                               in_bank[:, None] & (d[None, :] < DIM), other=0)
                kc += part.to(tl.float32)

        kv = tl.where((col < SWA)[:, None], ko, kc)
        valid = tl.where(col < SWA, oi >= 0, (col < SWA + TOPK) & (ci >= 0))
        maximum0, denominator0, accumulator0 = _update_head(
            q0, kv, valid, maximum0, denominator0, accumulator0, SCALE,
        )
        maximum1, denominator1, accumulator1 = _update_head(
            q1, kv, valid, maximum1, denominator1, accumulator1, SCALE,
        )

    result0 = accumulator0 / tl.where(denominator0 > 0, denominator0, 1.0)
    result1 = accumulator1 / tl.where(denominator1 > 0, denominator1, 1.0)
    tl.store(OUT + row * OUT_ROW + head * DIM + d, result0, d < DIM)
    tl.store(OUT + row * OUT_ROW + (head + 1) * DIM + d, result1, d < DIM)


def sparse_shared_kv_two_heads(q, original_kv, compressed_kv, original_indices, compressed_indices, sink, scale=None):
    assert q.dtype == original_kv.dtype == compressed_kv.dtype == torch.bfloat16
    assert q.ndim == 3 and q.shape[-1] == 512 and q.shape[1] in (8, 16, 32, 64)
    assert original_indices.ndim == 2 and original_indices.shape[0] == q.shape[0]
    assert original_indices.shape[1] >= 128 and compressed_indices.shape == (q.shape[0], 512)
    assert original_indices.dtype == compressed_indices.dtype == torch.int32
    assert sink.dtype == torch.float32 and sink.shape == (q.shape[1],)
    tensors = (q, original_kv, compressed_kv, original_indices, compressed_indices, sink)
    assert all(item.is_contiguous() and item.device == q.device for item in tensors)

    tables = []
    for pool in (original_kv, compressed_kv):
        prepared = getattr(pool, '_ds41_attention_banks', None)
        if prepared is None:
            raise RuntimeError('Call prepare_pool on the persistent pool views before capture')
        assert prepared[0] == (pool.data_ptr(), tuple(pool.shape), tuple(pool.stride()), str(pool.device))
        tables.append(prepared[1])

    result = torch.empty_like(q)
    if q.shape[0]:
        _sparse_shared_kv_two_heads[(q.shape[0], q.shape[1] // 2)](
            q, *tables, original_indices, compressed_indices, sink, result,
            q.stride(0), original_kv.stride(0), compressed_kv.stride(0),
            original_indices.stride(0), compressed_indices.stride(0), result.stride(0),
            tables[0].numel(), tables[1].numel(), BANK_ROWS, 512,
            original_indices.shape[1], 512, float(scale or 512**-0.5), 512, 8,
        )
    return result
