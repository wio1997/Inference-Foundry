"""Fixed-shape target attention metadata owned by the Extreme decode loop.

The bootstrap may lend tensors and operator callables.  This module retains no
scheduler, ModelRunner, attention builder, or request object.  It updates the
stable addresses consumed by the bound target for the c12 x 8-token TP8 path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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
        self._static_kv_max = os.getenv("EXTREME_TARGET_METADATA_STATIC_KV_MAX") == "1"
        self._shadow_dynamic = os.getenv("EXTREME_TARGET_METADATA_SHADOW") == "1"
        self._shadow_checks = 0
        self._shadow_stats: dict[str, dict[str, int]] = {}
        if self._shadow_dynamic and not self._static_kv_max:
            raise ValueError("metadata shadow requires the static KV bound")
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

    def make_scratch(self) -> "FixedTargetMetadataUpdater":
        """Create private outputs; immutable caches and block tables remain shared."""
        def private_prefix(tensor: torch.Tensor, count: int) -> torch.Tensor:
            return torch.empty_like(tensor[:count])

        groups = tuple(CPGroupBinding(
            ratio=group.ratio,
            seq_lens=private_prefix(group.seq_lens, 12),
            input_positions=private_prefix(group.input_positions, 96),
            start_pos=private_prefix(group.start_pos, 12),
            local_query_start_loc=private_prefix(group.local_query_start_loc, 13),
            local_seq_lens=private_prefix(group.local_seq_lens, 12),
            sas_metadata=private_prefix(group.sas_metadata, 1024),
            qli_metadata=(None if group.qli_metadata is None else
                          private_prefix(group.qli_metadata, 1024)),
            swa_slot_mapping=(None if group.swa_slot_mapping is None else
                              private_prefix(group.swa_slot_mapping, 96)),
            swa_block_table=group.swa_block_table,
            swa_block_size=group.swa_block_size,
        ) for group in self.groups)
        rotary = tuple(RotaryBinding(
            full_cos=binding.full_cos,
            full_sin=binding.full_sin,
            target_cos=private_prefix(binding.target_cos, 96),
            target_sin=private_prefix(binding.target_sin, 96),
            local_cos=(None if binding.local_cos is None else
                       torch.empty_like(binding.local_cos)),
            local_sin=(None if binding.local_sin is None else
                       torch.empty_like(binding.local_sin)),
        ) for binding in self.rotary)
        scratch = FixedTargetMetadataUpdater(
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
            query_start_loc=self._query_start_loc,
            groups=groups,
            rotary=rotary,
            operators=self.ops,
        )
        if scratch._static_kv_max != self._static_kv_max:
            raise RuntimeError("scratch KV max mode differs from active target")
        return scratch

    def commit_from(self, scratch: "FixedTargetMetadataUpdater") -> None:
        """Copy finished private results to stable Graph-addressed destinations."""
        if len(self.groups) != len(scratch.groups) or len(self.rotary) != len(scratch.rotary):
            raise RuntimeError("scratch metadata binding count changed")
        self.start_pos.copy_(scratch.start_pos)
        self.local_seq_lens.copy_(scratch.local_seq_lens)
        for active, private in zip(self.rotary, scratch.rotary):
            active.target_cos[:96].copy_(private.target_cos[:96])
            active.target_sin[:96].copy_(private.target_sin[:96])
            if active.local_cos is not None:
                if private.local_cos is None or private.local_sin is None:
                    raise RuntimeError("scratch local RoPE storage missing")
                active.local_cos.copy_(private.local_cos)
                active.local_sin.copy_(private.local_sin)
        for active, private in zip(self.groups, scratch.groups):
            if active.ratio != private.ratio:
                raise RuntimeError("scratch DSA ratio changed")
            for name, count in (("seq_lens",12), ("input_positions",96),
                                ("start_pos",12), ("local_query_start_loc",13),
                                ("local_seq_lens",12), ("sas_metadata",1024),
                                ("qli_metadata",1024), ("swa_slot_mapping",96)):
                destination = getattr(active, name)
                source = getattr(private, name)
                if destination is None:
                    if source is not None:
                        raise RuntimeError(f"unexpected scratch {name}")
                    continue
                if source is None:
                    raise RuntimeError(f"missing scratch {name}")
                destination[:count].copy_(source[:count])

    def _audit_shadow(self, name: str, static: torch.Tensor,
                      dynamic_a: torch.Tensor, dynamic_c: torch.Tensor,
                      stable_header: int) -> None:
        # The AICPU operators can leave their tail non-deterministic even on
        # identical inputs. A/C quantifies that floor; the stable header must
        # match exactly on every continuous cycle.
        a = dynamic_a[:1024]
        b = static[:1024]
        c = dynamic_c[:1024]
        ab = a != b
        ac = a != c
        if bool(ab[:stable_header].any().item()) or bool(ac[:stable_header].any().item()):
            raise AssertionError(f"{name} stable metadata header mismatch")
        stats = self._shadow_stats.setdefault(name, {
            "cycles": 0, "ab_diff": 0, "ac_diff": 0,
            "b_only_diff": 0, "first_ab": 1024, "first_ac": 1024,
        })
        ab_indices = torch.nonzero(ab).flatten()
        ac_indices = torch.nonzero(ac).flatten()
        stats["cycles"] += 1
        stats["ab_diff"] += int(ab_indices.numel())
        stats["ac_diff"] += int(ac_indices.numel())
        stats["b_only_diff"] += int((ab & ~ac & (b != c)).sum().item())
        if ab_indices.numel():
            stats["first_ab"] = min(stats["first_ab"], int(ab_indices[0].item()))
        if ac_indices.numel():
            stats["first_ac"] = min(stats["first_ac"], int(ac_indices[0].item()))

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
        # Both operators receive actual KV lengths in this fixed path. Their
        # scalar max is a fallback only when the length tensor is absent.
        # Keep the dynamic value for an opt-in exact shadow, never on the
        # measured static path.
        dynamic_kv_max = (
            max(1, int(self.local_seq_lens.max().item()))
            if not self._static_kv_max or self._shadow_dynamic else None
        )
        max_local_seq_len = 1 if self._static_kv_max else dynamic_kv_max
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
                sas_args = dict(
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
                sas = self.ops.sas(**sas_args)
                if self._shadow_dynamic:
                    sas_args["max_seqlen_kv"] = dynamic_kv_max
                    sas_dynamic_a = self.ops.sas(**sas_args)
                    sas_dynamic_c = self.ops.sas(**sas_args)
                    self._audit_shadow(f"sas_ratio_{ratio}", sas,
                                       sas_dynamic_a, sas_dynamic_c, 97)
                group.sas_metadata[:1024].copy_(sas[:1024])
                seen_sas.add(ratio)
            else:
                source = next(g.sas_metadata for g in self.groups if g.ratio == ratio)
                group.sas_metadata[:1024].copy_(source[:1024])
            if ratio == 4 and group.qli_metadata is not None:
                if ratio not in seen_qli:
                    qli_args = dict(
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
                    qli = self.ops.qli(**qli_args)
                    if self._shadow_dynamic:
                        qli_args["max_seqlen_k"] = dynamic_kv_max
                        qli_dynamic_a = self.ops.qli(**qli_args)
                        qli_dynamic_c = self.ops.qli(**qli_args)
                        self._audit_shadow("qli", qli,
                                           qli_dynamic_a, qli_dynamic_c, 25)
                    group.qli_metadata[:1024].copy_(qli[:1024])
                    seen_qli.add(ratio)
                else:
                    source = next(g.qli_metadata for g in self.groups if g.ratio == ratio)
                    group.qli_metadata[:1024].copy_(source[:1024])
        if self._shadow_dynamic:
            self._shadow_checks += 1
            if self._shadow_checks in (1, 64, 128, 256):
                print(
                    f"EXTREME_METADATA_SHADOW rank={self._tp_rank} "
                    f"checks={self._shadow_checks} pass=1 "
                    f"stats={self._shadow_stats}",
                    flush=True,
                )
