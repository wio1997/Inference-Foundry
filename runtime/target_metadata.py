"""Fixed-shape target attention metadata owned by the Extreme decode loop.

The bootstrap may lend tensors and operator callables.  This module retains no
scheduler, ModelRunner, attention builder, or request object.  It updates the
stable addresses consumed by the bound target for the c12 x 8-token TP8 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .fixed_decode import FixedDecodeState


@dataclass(frozen=True)
class RotaryBinding:
    full_cos: torch.Tensor
    full_sin: torch.Tensor
    target_cos: torch.Tensor
    target_sin: torch.Tensor
    local_cos: torch.Tensor | None = None
    local_sin: torch.Tensor | None = None


@dataclass(frozen=True)
class CPGroupBinding:
    ratio: int
    seq_lens: torch.Tensor
    input_positions: torch.Tensor
    start_pos: torch.Tensor
    local_query_start_loc: torch.Tensor
    local_seq_lens: torch.Tensor
    sas_metadata: torch.Tensor
    qli_metadata: torch.Tensor | None = None
    swa_slot_mapping: torch.Tensor | None = None
    swa_block_table: torch.Tensor | None = None
    swa_block_size: int | None = None


@dataclass(frozen=True)
class SparseMetadataOperators:
    sas: Callable[..., torch.Tensor]
    qli: Callable[..., torch.Tensor]
    cu_seqlens_ori_kv: torch.Tensor
    cu_seqlens_cmp_kv: torch.Tensor
    seqused_q: torch.Tensor
    device_name: str
    num_heads: int
    head_dim: int
    sliding_window: int
    index_topk: int
    index_n_heads: int
    index_head_dim: int


class FixedTargetMetadataUpdater:
    """Refresh fixed CP/DSA target tensors before every target forward."""

    def __init__(
        self,
        *,
        tp_rank: int,
        tp_size: int,
        query_start_loc: torch.Tensor,
        groups: tuple[CPGroupBinding, ...],
        rotary: tuple[RotaryBinding, ...],
        operators: SparseMetadataOperators,
    ) -> None:
        if not 0 <= tp_rank < tp_size:
            raise ValueError("invalid TP rank")
        if query_start_loc.ndim != 1 or query_start_loc.numel() != 13:
            raise ValueError("fixed c12 query_start_loc must have 13 elements")
        if not groups:
            raise ValueError("at least one CP target group is required")
        self.groups = groups
        self.rotary = rotary
        self.ops = operators
        self.batch_size = 12
        self.target_tokens = 96
        self._query_start_loc = query_start_loc
        self._device = query_start_loc.device
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        tokens_per_rank = (self.target_tokens + tp_size - 1) // tp_size
        self.local_start = tp_rank * tokens_per_rank
        self.local_end = self.local_start + tokens_per_rank
        local_begin = query_start_loc[:-1].clamp(self.local_start, self.local_end)
        local_end = query_start_loc[1:].clamp(self.local_start, self.local_end)
        local_lens = local_end - local_begin
        self.local_query_start_loc = torch.cat(
            (torch.zeros(1, dtype=query_start_loc.dtype, device=self._device),
             local_lens.cumsum(0, dtype=query_start_loc.dtype))
        )
        self._local_active = local_lens > 0
        self._local_offset = query_start_loc[1:] - local_end
        self.max_local_query_len = max(1, int(local_lens.max().item()))
        self.local_seq_lens = torch.empty(12, dtype=query_start_loc.dtype, device=self._device)
        self.start_pos = torch.empty(12, dtype=query_start_loc.dtype, device=self._device)
        self._request_index = torch.arange(12, device=self._device).unsqueeze(1)

    @torch.inference_mode()
    def update(self, state: FixedDecodeState) -> None:
        if state.target_positions.numel() != self.target_tokens:
            raise ValueError("fixed target must contain 96 positions")
        if state.target_seq_lens.numel() != self.batch_size:
            raise ValueError("fixed target must contain 12 sequence lengths")
        self.start_pos.copy_(state.target_seq_lens - 8)
        self.local_seq_lens.copy_(torch.where(
            self._local_active,
            (state.target_seq_lens - self._local_offset).clamp_min(0),
            torch.zeros_like(state.target_seq_lens),
        ))
        max_local_seq_len = max(1, int(self.local_seq_lens.max().item()))
        for binding in self.rotary:
            cos = binding.full_cos.index_select(0, state.target_positions.long())
            sin = binding.full_sin.index_select(0, state.target_positions.long())
            binding.target_cos[:96].copy_(cos)
            binding.target_sin[:96].copy_(sin)
            if binding.local_cos is not None:
                binding.local_cos.copy_(cos[self.local_start:self.local_end])
            if binding.local_sin is not None:
                binding.local_sin.copy_(sin[self.local_start:self.local_end])
        seen_sas: set[int] = set()
        seen_qli: set[int] = set()
        for group in self.groups:
            group.seq_lens[:12].copy_(state.target_seq_lens)
            group.input_positions[:96].copy_(state.target_positions)
            group.start_pos[:12].copy_(self.start_pos)
            group.local_query_start_loc[:13].copy_(self.local_query_start_loc)
            group.local_seq_lens[:12].copy_(self.local_seq_lens)
            if group.swa_slot_mapping is not None:
                if group.swa_block_table is None or group.swa_block_size is None:
                    raise ValueError("SWA slot binding needs block table and size")
                block_size = group.swa_block_size
                positions = state.target_positions.view(12, 8)
                block_index = torch.div(positions, block_size, rounding_mode="floor")
                blocks = group.swa_block_table[self._request_index, block_index]
                raw = (blocks * block_size + positions.remainder(block_size)).flatten()
                formatted = torch.stack((raw // block_size, raw % block_size), dim=-1)
                group.swa_slot_mapping[:96].copy_(formatted.to(group.swa_slot_mapping.dtype))
            ratio = group.ratio
            if ratio not in seen_sas:
                sas = self.ops.sas(
                    device=self.ops.device_name,
                    num_heads_q=self.ops.num_heads,
                    num_heads_kv=1,
                    head_dim=self.ops.head_dim,
                    cu_seqlens_q=self.local_query_start_loc,
                    cu_seqlens_ori_kv=self.ops.cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=self.ops.cu_seqlens_cmp_kv,
                    seqused_q=self.ops.seqused_q,
                    seqused_kv=self.local_seq_lens,
                    max_seqlen_q=self.max_local_query_len,
                    max_seqlen_kv=max_local_seq_len,
                    batch_size=12,
                    ori_mask_mode=4,
                    ori_win_left=self.ops.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    cmp_ratio=ratio,
                    has_cmp_kv=ratio > 1,
                    **({"cmp_mask_mode": 3} if ratio > 1 else {}),
                    **({"cmp_topk": self.ops.index_topk} if ratio == 4 else {}),
                )
                group.sas_metadata[:1024].copy_(sas[:1024])
                seen_sas.add(ratio)
            else:
                source = next(g.sas_metadata for g in self.groups if g.ratio == ratio)
                group.sas_metadata[:1024].copy_(source[:1024])
            if ratio == 4 and group.qli_metadata is not None:
                if ratio not in seen_qli:
                    qli = self.ops.qli(
                        actual_seq_lengths_query=self.local_query_start_loc[1:].clone(),
                        actual_seq_lengths_key=self.local_seq_lens.clone(),
                        num_heads_q=self.ops.index_n_heads,
                        num_heads_k=1,
                        head_dim=self.ops.index_head_dim,
                        query_quant_mode=0,
                        key_quant_mode=0,
                        batch_size=12,
                        max_seqlen_q=self.max_local_query_len,
                        max_seqlen_k=max_local_seq_len,
                        layout_query="TND",
                        layout_key="PA_BSND",
                        sparse_count=self.ops.index_topk,
                        sparse_mode=3,
                        pre_tokens=(1 << 63) - 1,
                        next_tokens=(1 << 63) - 1,
                        cmp_ratio=4,
                        device=self.ops.device_name,
                    )
                    group.qli_metadata[:1024].copy_(qli[:1024])
                    seen_qli.add(ratio)
                else:
                    source = next(g.qli_metadata for g in self.groups if g.ratio == ratio)
                    group.qli_metadata[:1024].copy_(source[:1024])
