# SPDX-License-Identifier: Apache-2.0
"""Flash-only bounded eager KDA reference using B103's accurate recurrent core.

R051 qualifies this native core's internal FP32 norm/gate/state against CPU.
No new weights/operators/dependencies. Native conv and decode stay inherited.
CPU metadata checks are diagnostic; graph/MTP/state tracking need later work.
"""
from contextvars import ContextVar
import os
import torch
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import AscendKDAAttnBackend

_INITIAL = ContextVar('flash_kda_initial_mask', default=None)
_TRACK = ContextVar('flash_kda_track_chunk', default=None)
_PERFORMANCE = os.environ.get('FLASH_PD_PERF_MODE', '0') == '1'


def accurate_flash_extend(q, k, v, raw_gate, raw_beta, *, ssm_states,
                          cache_indices, query_start_loc, has_initial_state,
                          A_log, dt_bias, lower_bound, track_state=None, track_chunk_idx=None):
    if q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape or v.shape != q.shape:
        raise ValueError('Flash reference requires [1,actual_tokens,heads,128] Q/K/V')
    total, heads, dim = q.shape[1:]
    if dim != 128 or lower_bound != -5.0 or total > 4096:
        raise ValueError('Initial Flash reference requires dim128/gate-5/at most4096tokens')
    if any(x.dtype != torch.bfloat16 for x in [q,k,v,raw_gate,raw_beta]):
        raise TypeError('Qualified Flash reference inputs are BF16')
    if ssm_states.dtype != torch.float32 or ssm_states.shape[1:] != (heads,128,128):
        raise TypeError('Flash persistent state must be FP32 canonical VK')
    if A_log.numel() != heads or dt_bias.numel() != heads*dim:
        raise ValueError('Flash gate parameter shape mismatch')
    if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise TypeError('Flash A_log/dt_bias must preserve FP32 checkpoint tensors')
    if raw_gate.numel() != total*heads*dim or raw_beta.numel() != total*heads:
        raise ValueError('Flash raw projection token/head shape mismatch')
    if any(x.device != q.device for x in [k,v,raw_gate,raw_beta,ssm_states,A_log,dt_bias,cache_indices,query_start_loc]):
        raise ValueError('Flash kernel inputs and persistent state must share a device')
    if cache_indices.dtype not in [torch.int32,torch.int64] or query_start_loc.dtype not in [torch.int32,torch.int64]:
        raise TypeError('Flash slot and boundary metadata must be integral')
    if has_initial_state.dtype != torch.bool:
        raise TypeError('Flash fresh-prefix mask must be boolean')
    if not _PERFORMANCE:
        ends = query_start_loc.detach().cpu().tolist()
        slots = cache_indices.detach().cpu().reshape(-1).tolist()
        initial = has_initial_state.detach().cpu().reshape(-1).tolist()
        if len(ends) != len(slots)+1 or len(initial) != len(slots) or not ends or ends[0] != 0 or ends[-1] != total:
            raise ValueError('Flash request boundaries/slot/initial-mask mismatch')
        if any(a>b for a,b in zip(ends,ends[1:])):
            raise ValueError('Flash request boundaries must be nondecreasing')
        active=[]
        for i,slot in enumerate(slots):
            if ends[i] == ends[i+1]:continue
            if not 0 < slot < ssm_states.shape[0]:
                raise ValueError('Active Flash sequence requires a non-NULL state slot')
            active.append(slot)
        if len(set(active)) != len(active):raise ValueError('Duplicate active Flash state slots')
        if total == 0:return v.new_empty(v.shape)
        # Validate everything before touching persistent state; fresh requests never
        # inherit a recycled owner's recurrence. Empty/padded requests do not clear.
        for i,slot in enumerate(slots):
            if ends[i] < ends[i+1] and not initial[i]:ssm_states[slot].zero_()
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log.reshape(-1).contiguous(),
        a=raw_gate.reshape(1,total,heads,dim).contiguous(),
        dt_bias=dt_bias.reshape(-1).contiguous(),softplus_beta=1.,softplus_threshold=20.,
        q=q,k=k,v=v,b=raw_beta.reshape(1,total,heads).contiguous(),
        initial_state_source=ssm_states,initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,scale=dim**-.5,
        use_qk_l2norm_in_kernel=True,is_kda=True,lower_bound=lower_bound,
        initial_state_mask=has_initial_state,track_state=track_state,track_chunk_idx=track_chunk_idx,
    )


class FlashKDAExtendKernel:
    supports_track_state_snapshot = True

    def extend(self,q,k,v,g,beta,*,ssm_states,cache_indices,query_start_loc,
               return_intermediate_states=False,**kwargs):
        initial=_INITIAL.get()
        if initial is None:raise ValueError('Missing actual Flash fresh-prefix metadata')
        track_chunk_idx=_TRACK.get() if return_intermediate_states else None
        h=None
        if track_chunk_idx is not None:
            h=torch.empty((query_start_loc.shape[0]-1,*ssm_states.shape[1:]),
                dtype=torch.float32,device=ssm_states.device)
        output=accurate_flash_extend(q,k,v,g,beta,ssm_states=ssm_states,
            cache_indices=cache_indices,query_start_loc=query_start_loc,
            has_initial_state=initial,A_log=kwargs['A_log'],dt_bias=kwargs['dt_bias'],
            lower_bound=kwargs['lower_bound'],track_state=h,track_chunk_idx=track_chunk_idx)
        return (output,h) if return_intermediate_states else output


class FlashAscendKDAAttnBackend(AscendKDAAttnBackend):
    def __init__(self,model_runner):
        super().__init__(model_runner)
        self.kernel_dispatcher.extend_kernel=FlashKDAExtendKernel()

    def _prepare_extend_gate_inputs(self,layer,g,beta):
        # The native chunk path expects activated gate and beta coefficients;
        # the accurate recurrent core instead consumes both raw projections.
        return g,beta,layer.A_log,layer.dt_bias

    def forward_extend(self,layer,forward_batch,mixed_qkv,a,b,**kwargs):
        if forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend_v2():
            raise ValueError('Flash MTP state snapshots/rollback are not qualified yet')
        prefix=forward_batch.extend_prefix_lens
        if prefix is None:raise ValueError('Missing actual Flash prefix lengths')
        token=_INITIAL.set(prefix>0)
        tracked=_TRACK.set(getattr(self.forward_metadata,"track_chunk_idx",None))
        from sglang.srt.managers.flash_pd_request_context import forward_tags
        from sglang.srt.hardware_backend.npu.attention.flash_b103_pd_state_audit import snapshot
        active=None if _PERFORMANCE else forward_tags()
        try:
            from sglang.srt.hardware_backend.npu.attention.flash_kda_prefix_once_audit import before as _prefix_before, after as _prefix_after
            _prefix_before(self,layer,forward_batch)
            if not _PERFORMANCE:snapshot(self,layer,forward_batch,'extend_before',active)
            output=super().forward_extend(layer,forward_batch,mixed_qkv,a,b,**kwargs)
            _prefix_after(self,layer,forward_batch)
            if not _PERFORMANCE:snapshot(self,layer,forward_batch,'extend_after',active)
            return output
        finally:
            _TRACK.reset(tracked)
            _INITIAL.reset(token)

    def _track_mamba_state_extend(self,forward_batch,h,ssm_states,forward_metadata,
                                  track_states=None,*,h_track_buf=None):
        # Ascend extend receives our compact FP32 per-batch checkpoint. Native
        # tracking copies only selected rows and retains its aligned-final path.
        return super()._track_mamba_state_extend(forward_batch,None,ssm_states,
            forward_metadata,track_states,h_track_buf=h)

    def forward_decode(self,layer,forward_batch,mixed_qkv,a,b,**kwargs):
        from sglang.srt.managers.flash_pd_request_context import forward_tags
        from sglang.srt.hardware_backend.npu.attention.flash_b103_pd_state_audit import snapshot
        active=None if _PERFORMANCE else forward_tags()
        if not _PERFORMANCE:snapshot(self,layer,forward_batch,'decode_before',active)
        output=super().forward_decode(layer,forward_batch,mixed_qkv,a,b,**kwargs)
        if not _PERFORMANCE:snapshot(self,layer,forward_batch,'decode_after',active)
        return output
