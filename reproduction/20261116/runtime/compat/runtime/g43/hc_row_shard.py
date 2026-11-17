"""Opt-in A3 target-model HC row sharding; original attention metadata/math.

This is an unvalidated candidate. It reuses attention-TP all-gather and stock
AscendOps transport, keeping FP32 HC statistics and full attention reductions.
"""
from dataclasses import dataclass
import logging
import os

import torch

ENABLED = os.environ.get("SGLANG_NPU_MHC_ROW_SHARD", "0") == "1"
_LOGGED = set()


@dataclass(frozen=True)
class RowLayout:
    rows: int
    size: int
    rank: int

    def __post_init__(self):
        assert self.rows >= 0 and self.size > 0 and 0 <= self.rank < self.size

    @property
    def chunk(self):
        return (self.rows + self.size - 1) // self.size

    @property
    def start(self):
        return self.rank * self.chunk

    @property
    def count(self):
        return max(0, min(self.chunk, self.rows - self.start))

    def shard(self, full):
        assert full.shape[0] == self.rows, (full.shape, self)
        # Own storage: a contiguous slice alone would retain the full residual.
        local = full.new_zeros((self.chunk, *full.shape[1:]))
        if self.count:
            local[:self.count].copy_(full[self.start:self.start + self.count])
        return local

    def gather(self, local):
        from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor

        assert local.shape[0] == self.chunk, (local.shape, self)
        if self.chunk == 0:
            return local
        full = local.new_empty((self.chunk * self.size, *local.shape[1:]))
        attn_tp_all_gather_into_tensor(full, local.contiguous())
        # Padding occurs once at the global tail, never between rank chunks.
        return full[:self.rows]

    def map_state(self, state, fn):
        from sglang.srt.models import deepseek_v4_mhc as mhc

        assert isinstance(state.streams, torch.Tensor)
        assert state.input is None and state.stats is None, "Unsupported HC cached state"
        return mhc.HcState(fn(state.residual), None if state.pre is None else fn(state.pre))

    def shard_state(self, state):
        return self.map_state(state, self.shard)

    def gather_state(self, state):
        return self.map_state(state, self.gather)

    def resized(self, rows):
        return type(self)(rows, self.size, self.rank)


def prepare(model, forward_batch, rows):
    """Reject unsupported layouts before any candidate collective is issued."""
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_forward, get_parallel

    p = get_parallel()
    assert not is_cp_active(forward_batch) and not get_forward().sp_active
    assert p.attn_cp_size == 1 and p.attn_tp_size == 4 and p.attn_dp_size == 4
    assert p.tp_size == 16 and p.moe_ep_size == 16
    assert get_moe_a2a_backend().is_ascend_ops()
    assert model.pp_group.world_size == 1
    assert model.decoder_replay_tail is None, "Tail graph contract not yet validated"
    assert not getattr(forward_batch, 'tbo_children', None)
    assert model.config.model_type == 'deepseek_v41'
    layout = RowLayout(rows, p.attn_tp_size, p.attn_tp_rank)
    key = (p.tp_rank, 'prepare')
    if key not in _LOGGED:
        _LOGGED.add(key)
        logging.getLogger(__name__).warning(
            'A3_V41_HC_ROW_SHARD rank=%s tp=%s dp=%s full_attention=True fp32_stats=True',
            p.tp_rank, p.attn_tp_size, p.attn_dp_size)
    return layout


def forward_layer(layer, *, positions, state, input_ids, forward_batch,
                  input_ids_global, layout):
    """Keep HC/MoE local, gather collapsed H before unchanged full attention."""
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport

    assert isinstance(state.streams, torch.Tensor) and state.residual.device.type == 'npu'
    assert state.input is None and state.stats is None
    assert not layer.self_attn.accepts_mxfp8_swizzled_input()
    assert layer.mlp.shared_experts is None or layer.mlp._shared_expert_tp1
    assert positions.shape[0] == layout.rows
    layer._init_boundaries()
    stats_stream = (layer.hc_stats_stream
                    if mhc.use_stats_stream(layer.hc_cfg, forward_batch, state.residual)
                    else None)
    residual = state.residual
    mhc.fork_stats_stream(stats_stream)
    x = layout.gather(mhc.combine(layer.attn_hc, state))
    state.release()
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        y = layer.self_attn(x=x, positions=positions, forward_batch=forward_batch,
                            x_quant=None, defer_all_reduce=False)
    del x
    assert isinstance(y, torch.Tensor) and y.shape[0] == layout.rows
    y_local = layout.shard(y)
    del y
    state = mhc.run_attn_post(layer.attn_hc, y_local, residual,
                             stats_stream=stats_stream, next=layer.local_boundary,
                             world_size=layer.self_attn.attn_tp_size)
    del residual, y_local

    residual = state.residual
    mhc.fork_stats_stream(stats_stream)
    x = mhc.combine(layer.ffn_hc, state)
    assert state.input is None and state.stats is None
    state.release()
    ids = None if input_ids is None else layout.shard(input_ids)
    global_ids = None if input_ids_global is None else layout.shard(input_ids_global)
    # Existing explicit overrides prevent a second scatter and output gather.
    prepared = MoeRowTransport(layer.mlp, x, forward_batch, ids, global_ids,
                               use_cp=False, moe_gather=False, attn_scatter=False)
    with prepared.scope():
        y = layer.mlp(prepared.input, forward_batch, input_ids=prepared.input_ids,
                      input_ids_global=prepared.input_ids_global,
                      skip_shared_experts=prepared.do_shared_local,
                      return_moe_output=False)
    prepared.input = None
    y = prepared.return_output(y)
    assert isinstance(y, torch.Tensor) and y.shape[0] == layout.chunk
    del x, prepared, ids, global_ids
    return mhc.run_moe_post(layer.ffn_hc, y, residual,
                            stats_stream=stats_stream, next=layer.next_boundary,
                            world_size=layer.mlp.tp_size)
