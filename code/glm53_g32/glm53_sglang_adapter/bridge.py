# SPDX-License-Identifier: Apache-2.0
"""Explicit SGLang physical-token addressing -> Ascend donor KPool tensors.

This module imports no vLLM. Storage owners must supply registered cache tensors.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from .vendor.sparse_attn_indexer_kpool import SparseAttnIndexerKpool


@dataclass
class KPoolStorage:
    pooled: torch.Tensor  # [physical raw pages, raw_page_size/R, 1, 128], BF16
    tail: torch.Tensor    # [request slots, 2, R+lookahead, 128], FP32
    raw_page_size: int
    ratio: int = 4

    def validate(self):
        if self.raw_page_size % self.ratio or self.ratio != 4:
            raise ValueError('This GLM snapshot requires raw pages divisible by index_kpool=4')
        if self.pooled.ndim != 4 or self.pooled.shape[1:] != (self.raw_page_size // self.ratio, 1, 128):
            raise ValueError('Pooled cache page layout does not match physical raw pages')
        if self.pooled.dtype != torch.bfloat16:
            raise TypeError('Ascend donor requires BF16 compressed index keys')
        if self.tail.ndim != 4 or self.tail.shape[1] != 2 or self.tail.shape[2] < self.ratio or self.tail.shape[3] != 128:
            raise ValueError('Tail must retain [request_slots, 2, R+lookahead, 128]')
        if self.tail.dtype != torch.float32 or self.tail.device != self.pooled.device:
            raise TypeError('Tail requires FP32 on the pooled-cache device')


def map_completed_slots(raw_slots, positions, raw_page_size, ratio=4):
    if raw_page_size % ratio:
        raise ValueError('Raw page size must be divisible by ratio')
    # Equivalent to block_id*(raw_page_size/R)+offset//R. Dummy rows stay -1.
    good = (raw_slots >= 0) & (positions >= 0) & ((positions + 1) % ratio == 0)
    return torch.where(good, raw_slots.clamp_min(0) // ratio, torch.full_like(raw_slots, -1))


def build_metadata(storage, raw_slots, positions, raw_seq_lens, query_lens, raw_page_table, req_pool_indices):
    storage.validate()
    if raw_seq_lens.shape != query_lens.shape or query_lens.shape != req_pool_indices.shape:
        raise ValueError('Per-request lengths and request-slot indices must have identical shapes')
    if raw_page_table.shape[0] != query_lens.numel() or positions.shape != raw_slots.shape:
        raise ValueError('Request table/token slot dimensions disagree')
    ends = query_lens.to(torch.int32).cumsum(0).to(torch.int32)
    # Repeat by explicit token count; no CPU sequence copies or per-token Python loops.
    token_req_slots = torch.repeat_interleave(req_pool_indices, query_lens.long(), output_size=positions.numel())
    capacity = storage.tail.shape[2]
    tail_slots = token_req_slots.long() * capacity + positions.long().remainder(capacity)
    tail_slots = torch.where(positions >= 0, tail_slots, torch.full_like(tail_slots, -1))
    meta = SimpleNamespace(
        block_table=raw_page_table.to(torch.int32).contiguous(),
        slot_mapping=map_completed_slots(raw_slots, positions, storage.raw_page_size, storage.ratio).long().contiguous(),
        seq_lens=(raw_seq_lens // storage.ratio).to(torch.int32).contiguous(),
        raw_seq_lens=raw_seq_lens.to(torch.int32).contiguous(),
        cum_query_lens=ends.contiguous(),
    )
    tail_meta = SimpleNamespace(slot_mapping=tail_slots.contiguous(), block_table=req_pool_indices.to(torch.int32)[:, None].contiguous(), block_size=capacity)
    return meta, tail_meta


def run_kpool(storage, *, key, query, head_weights, gate_score, ape, positions,
              raw_slots, raw_seq_lens, query_lens, raw_page_table, req_pool_indices,
              topk=2048, compute_topk=True, speculative_lookahead=0, query_positions=None):
    if speculative_lookahead < 0 or storage.tail.shape[2] < storage.ratio + speculative_lookahead:
        raise ValueError('Speculative tail retention is smaller than R+lookahead')
    meta, tail_meta = build_metadata(storage, raw_slots, positions, raw_seq_lens, query_lens, raw_page_table, req_pool_indices)
    op = SparseAttnIndexerKpool(topk, 128)
    query_meta=(SimpleNamespace(positions=query_positions,cum_query_lens=meta.cum_query_lens,num_actual_tokens=query_positions.numel())
                if query_positions is not None else None)
    return op(key.float(), query, head_weights, positions, storage.pooled, storage.tail, meta, tail_meta,
              gate_score=gate_score.float(), compress_ape=ape.float(), index_kpool=storage.ratio,
              max_pool_seq_len=raw_page_table.shape[1] * storage.pooled.shape[1],
              compute_topk=compute_topk, allow_cache_packing=True, query_metadata=query_meta)
