# SPDX-License-Identifier: Apache-2.0
"""GLM bounded-gate KDA with explicit per-request state.

The torch path is a correctness fallback, not a throughput implementation.
The AscendC path lazily uses fla_npu; it requires a separately validated runtime
(e.g. torch/torch_npu 2.10 + flash-linear-attention-npu-a3 26.9.1).
No cache slots, tensor parallel collectives, MTP or model registry are implied.
Equations/layout follow the GLM vLLM-Ascend KDA contract, with independent
q/k/v SiLU short convolutions and a value-first FP32 recurrent state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class KDAState:
    # Each conv stores the preceding W-1 *raw* projected tokens, oldest first.
    conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    recurrent: torch.Tensor  # [heads, value, key], FP32
    tokens_seen: int = 0


def causal_conv(x: torch.Tensor, weight: torch.Tensor, history: torch.Tensor):
    """[tokens, channels], [channels, width], [width-1, channels]."""
    if x.ndim != 2 or weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError('invalid causal convolution shape')
    width = weight.shape[1]
    if width < 1 or history.shape != (width - 1, x.shape[1]):
        raise ValueError('invalid causal convolution history')
    if history.dtype != x.dtype or history.device != x.device or weight.device != x.device:
        raise ValueError('convolution tensors must share device and history dtype')
    if not x.shape[0]:
        return x.clone(), history.clone()
    joined = torch.cat((history, x), dim=0)
    # FP32 accumulation; activation rounded once to the projection dtype.
    weight = weight.to(x.dtype)  # GLM merges/casts F32 checkpoint conv weights to BF16.
    out = torch.zeros_like(x, dtype=torch.float32)
    for j in range(width):
        out = out + joined[j:j + x.shape[0]].float() * weight[:, j].float()
    out = F.silu(out).to(x.dtype)
    tail = joined[-(width - 1):].clone() if width > 1 else joined[:0].clone()
    return out, tail


def ascendc_causal_conv(x, weight, history):
    """Native GLM SiLU convolution with private cache scratch for one request."""
    if x.device.type != 'npu' or x.dtype != torch.bfloat16:
        raise ValueError('AscendC GLM convolution requires NPU BF16')
    if x.ndim!=2 or weight.ndim!=2 or weight.shape[0]!=x.shape[1] or not 2<=weight.shape[1]<=4:
        raise ValueError('invalid AscendC GLM convolution shape')
    if history.shape!=(weight.shape[1]-1,x.shape[1]) or history.dtype!=x.dtype or history.device!=x.device:
        raise ValueError('invalid AscendC GLM convolution history')
    if x.shape[0]==0:
        return x.clone(),history.clone()
    from fla_npu.ops.ascendc import causal_conv1d_fn
    # Reserve slot zero, keep the caller's raw history immutable.
    cache=torch.stack((torch.zeros_like(history),history)).contiguous()
    out=causal_conv1d_fn(
        x.contiguous(),weight.T.to(x.dtype).contiguous(),None,conv_states=cache,
        query_start_loc=torch.tensor([0,x.shape[0]],device=x.device,dtype=torch.int32),
        cache_indices=torch.ones(1,device=x.device,dtype=torch.int32),
        has_initial_state=torch.ones(1,device=x.device,dtype=torch.bool),
        activation='silu',null_block_id=0,
    )
    return out,cache[1]


def recurrent_kda(q, k, v, raw_gate, raw_beta, state, a_log, dt_bias, *, lower_bound=-5.0):
    """One sequence [T,H,K]; raw beta [T,H]; returns output and new VK state.

    Normalize Q/K before the update, decay the *key* axis, then apply the delta
    correction. Input state is never mutated, so a caller can commit or discard
    a speculative branch explicitly (MTP slot management is not implemented).
    """
    if q.ndim != 3 or q.shape != k.shape or q.shape != raw_gate.shape:
        raise ValueError('expected matching Q/K/raw_gate [T,H,K]')
    t, h, d = q.shape
    if v.ndim != 3 or v.shape[:2] != (t, h) or raw_beta.shape != (t, h):
        raise ValueError('invalid V/beta shape')
    if state.shape != (h, v.shape[-1], d) or state.dtype != torch.float32:
        raise ValueError('state must be FP32 [H,V,K]')
    if a_log.numel() != h or dt_bias.numel() != h * d:
        raise ValueError('invalid bounded gate parameters')
    if not math.isfinite(lower_bound) or not -5 <= lower_bound < 0:
        raise ValueError('bounded GLM gate requires lower_bound in [-5,0)')
    if len({x.device for x in (q,k,v,raw_gate,raw_beta,state,a_log,dt_bias)}) != 1:
        raise ValueError('KDA tensors must share a device')
    qf, kf = q.float(), k.float()
    # Match GLM's chunk-prefill l2norm boundary: FP32 sum+1e-6, then
    # round normalized vectors to the activation dtype before recurrence.
    # Use the same boundary in decode to avoid a prefill/decode semantic split.
    qf = (qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype).float() * d ** -0.5
    kf = (kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype).float()
    gate = lower_bound * torch.sigmoid(a_log.float().reshape(h,1).exp() *
                                      (raw_gate.float() + dt_bias.float().reshape(h,d)))
    beta = raw_beta.float().sigmoid()
    current = state.clone()
    outputs = []
    for i in range(t):
        current = current * gate[i].exp().unsqueeze(-2)
        delta = (v[i].float() - (current * kf[i].unsqueeze(-2)).sum(-1)) * beta[i].unsqueeze(-1)
        current = current + delta.unsqueeze(-1) * kf[i].unsqueeze(-2)
        outputs.append((current * qf[i].unsqueeze(-2)).sum(-1).to(v.dtype))
    output = torch.stack(outputs) if outputs else v.new_empty((0,h,v.shape[-1]))
    return output, current


def ascendc_recurrent_kda(q, k, v, raw_gate, raw_beta, state, a_log, dt_bias, *, lower_bound=-5.0):
    """Native one-sequence adapter. Keeps caller state immutable; BF16 K=128."""
    if q.device.type != 'npu' or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError('AscendC GLM KDA requires NPU BF16 and head_dim=128')
    if q.shape != k.shape or q.shape != v.shape or q.shape != raw_gate.shape or raw_beta.shape != q.shape[:2]:
        raise ValueError('invalid GLM AscendC projection shapes')
    if state.shape != (q.shape[1],128,128) or state.dtype != torch.float32:
        raise ValueError('invalid GLM AscendC VK state')
    if a_log.numel() != q.shape[1] or dt_bias.numel() != q.shape[1]*128:
        raise ValueError('invalid GLM AscendC gate parameters')
    if not math.isfinite(lower_bound) or not -5 <= lower_bound < 0:
        raise ValueError('bounded GLM gate requires lower_bound in [-5,0)')
    if q.shape[0] == 0:
        return v.clone(), state.clone()
    # The recurrent operator accepts at most eight tokens per sequence. Keep
    # chunks bounded and carry the final VK state between calls.
    from fla_npu.ops.ascendc import recurrent_kda as native_kda
    # fla_npu 26.9.1 fused QK normalization does not reproduce sum+1e-6
    # for small GLM projections. Normalize explicitly, matching chunk prefill.
    qf, kf = q.float(), k.float()
    q = (qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
    k = (kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    current = torch.stack((torch.zeros_like(state), state)).contiguous()
    outputs = []
    for first in range(0,q.shape[0],8):
        last = min(first+8,q.shape[0])
        cu = torch.tensor([0,last-first], device=q.device, dtype=torch.int32)
        output, _ = native_kda(
            q[first:last].unsqueeze(0).contiguous(), k[first:last].unsqueeze(0).contiguous(),
            v[first:last].unsqueeze(0).contiguous(), raw_gate[first:last].unsqueeze(0).contiguous(),
            raw_beta[first:last].unsqueeze(0).contiguous(), current,
            cu_seqlens=cu, ssm_state_indices=torch.ones((last-first,), device=q.device, dtype=torch.int32),
            A_log=a_log.reshape(-1).float().contiguous(), dt_bias=dt_bias.reshape(-1).float().contiguous(),
            layout='BSND', scale=128**-0.5, output_final_state=False,
            inplace_final_state=True, state_v_first=True, use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=False, safe_gate=True, lower_bound=lower_bound,
        )
        outputs.append(output.squeeze(0))
    return torch.cat(outputs), current[1]


def installed_decode_recurrent_kda(q, k, v, raw_gate, raw_beta, state, a_log, dt_bias, *, lower_bound=-5.0):
    """Opt-in installed decode core: exactly one token, private state slot."""
    if q.device.type != 'npu' or q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError('installed decode KDA requires one NPU BF16 token, head_dim=128')
    if q.shape != k.shape or q.shape != v.shape or q.shape != raw_gate.shape or raw_beta.shape != q.shape[:2]:
        raise ValueError('invalid installed decode KDA projection shapes')
    if state.shape != (q.shape[1],128,128) or state.dtype != torch.float32:
        raise ValueError('invalid installed decode KDA state')
    if a_log.numel() != q.shape[1] or dt_bias.numel() != q.shape[1]*128 or not math.isfinite(lower_bound) or not -5 <= lower_bound < 0:
        raise ValueError('invalid installed decode KDA gate parameters')
    if len({x.device for x in (q,k,v,raw_gate,raw_beta,state,a_log,dt_bias)}) != 1:
        raise ValueError('installed decode KDA tensors must share a device')
    from vllm_ascend.ops.kda import run_recurrent_kda
    pool=torch.stack((torch.zeros_like(state),state)).contiguous()
    cu=torch.tensor([0,1],device=q.device,dtype=torch.int32)
    indices=torch.ones((1,),device=q.device,dtype=torch.int32)
    output=run_recurrent_kda(q.unsqueeze(0).contiguous(),k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),raw_gate.unsqueeze(0).contiguous(),raw_beta.unsqueeze(0).contiguous(),
        pool,cu,indices,a_log.float(),dt_bias.float(),lower_bound=lower_bound,beta_is_preprocessed=False)
    return output.squeeze(0),pool[1]


class Glm5NextKDA(nn.Module):
    """Single unsharded GLM KDA layer with actual checkpoint projection names."""
    @classmethod
    def from_checkpoint(cls, reader, layer_id, device, *, backend='torch'):
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        config = Glm5NextConfig.from_checkpoint(reader.root)
        if not 0 <= layer_id < config.num_hidden_layers or config.layers[layer_id].attention != 'linear_attention':
            raise ValueError('requested layer is not a GLM KDA backbone layer')
        if backend not in ('torch','ascendc','ascendc_installed_decode','ascendc_chunk_prefill'):
            raise ValueError('unknown KDA backend')
        self = cls()
        lc = config.linear_attn_config
        self.num_heads, self.head_dim = lc['num_heads'], lc['head_dim']
        self.hidden_size, self.conv_width = config.hidden_size, lc['short_conv_kernel_size']
        self.lower_bound, self.backend = lc['gate_lower_bound'], backend
        self.chunk_prefill_single = False  # explicit singleton prefill, not decode
        self.reference_sharded_projection = False  # experimental reference-shaped GEMMs
        self.norm_eps = 1e-5  # FusedRMSNormGated GLM default; independent of hc_eps.
        prefix = f'model.language_model.layers.{layer_id}.self_attn'
        for attr, suffix in [('q_proj','q_proj'),('k_proj','k_proj'),('v_proj','v_proj'),
                             ('b_proj','b_proj'),('f_a_proj','forget_gate.f_a_proj'),
                             ('f_b_proj','forget_gate.f_b_proj'),('g_a_proj','g_a_proj'),
                             ('g_b_proj','g_b_proj'),('o_proj','o_proj')]:
            if reader.quant.get(f'{prefix}.{suffix}.weight') != 'FLOAT':
                raise ValueError('this KDA contract requires declared FLOAT projections')
            setattr(self,attr,reader.load_linear(f'{prefix}.{suffix}',device))
        # Match the installed Ascend reference's six-way BF16 A GEMM.
        # Rebind the original module weights as views of the fused storage so
        # access to q/k/v remains available without doubling persistent HBM.
        self.in_proj_qkvbfg_a = None
        if backend in ('ascendc','ascendc_installed_decode','ascendc_chunk_prefill'):
            from deepinfer.hugging_face.modelslim910c import FloatLinear
            projections = (self.q_proj, self.k_proj, self.v_proj, self.b_proj, self.f_a_proj, self.g_a_proj)
            if all(type(m) is FloatLinear and m.weight.dtype == torch.bfloat16 for m in projections):
                sizes = tuple(m.weight.shape[0] for m in projections)
                fused = torch.cat(tuple(m.weight for m in projections), dim=0).contiguous()
                for module, view in zip(projections, fused.split(sizes, dim=0), strict=True):
                    module.weight = nn.Parameter(view, requires_grad=False)
                self.in_proj_qkvbfg_a = FloatLinear(fused)
        for attr, suffix in [('a_log','forget_gate.A_log'),('dt_bias','forget_gate.dt_bias'),
                             ('o_norm_weight','o_norm.weight'),('q_conv','q_conv1d.weight'),
                             ('k_conv','k_conv1d.weight'),('v_conv','v_conv1d.weight')]:
            value = reader.tensor(f'{prefix}.{suffix}')[0].to(device)
            self.register_buffer(attr,value)
        width = self.num_heads*self.head_dim
        if self.a_log.numel()!=self.num_heads or self.dt_bias.numel()!=width or self.o_norm_weight.shape!=(self.head_dim,):
            raise ValueError('unexpected GLM KDA gate/norm layout')
        for value in (self.q_conv,self.k_conv,self.v_conv):
            if value.shape!=(width,1,self.conv_width):
                raise ValueError('unexpected GLM short-convolution layout')
        return self.eval()

    def initial_state(self):
        exemplar = self.q_conv
        width = self.num_heads*self.head_dim
        dtype = self.q_proj.weight.dtype
        conv = tuple(torch.zeros((self.conv_width-1,width),dtype=dtype,device=exemplar.device) for _ in range(3))
        return KDAState(conv,torch.zeros((self.num_heads,self.head_dim,self.head_dim),dtype=torch.float32,device=exemplar.device))

    @torch.inference_mode()
    def forward(self, x, state=None):
        if x.ndim!=2 or x.shape[-1]!=self.hidden_size:
            raise ValueError('expected hidden states [tokens,hidden_size]')
        if self.backend == 'ascendc_chunk_prefill' and not 1 <= x.shape[0] <= 64:
            raise ValueError('experimental chunk KDA requires 1..64 tokens')
        state = self.initial_state() if state is None else state
        if self.in_proj_qkvbfg_a is None:
            q, k, v = (self.q_proj(x), self.k_proj(x), self.v_proj(x))
            raw_beta, f_a, g_a = self.b_proj(x), self.f_a_proj(x), self.g_a_proj(x)
        else:
            width = self.num_heads * self.head_dim
            sizes = (width, width, width, self.num_heads, self.head_dim, self.head_dim)
            if type(self.reference_sharded_projection) is not bool:
                raise ValueError('reference projection mode must be boolean')
            if self.reference_sharded_projection:
                if self.backend != 'ascendc_chunk_prefill' or width % 4 or self.num_heads % 4:
                    raise ValueError('reference projection requires compatible chunk KDA')
                # Match four rank-local GEMM shapes, preserving replicated f/g gates.
                weights = self.in_proj_qkvbfg_a.weight.split(sizes, dim=0)
                local_sizes = (width//4, width//4, width//4, self.num_heads//4,
                               self.head_dim, self.head_dim)
                parts = []
                for shard in range(4):
                    local_weight = torch.cat(tuple(
                        w[shard*(w.shape[0]//4):(shard+1)*(w.shape[0]//4)] if i < 4 else w
                        for i, w in enumerate(weights)), dim=0).contiguous()
                    parts.append(F.linear(x, local_weight).split(local_sizes, dim=-1))
                q, k, v, raw_beta, f_a, g_a = tuple(
                    torch.cat([part[i] for part in parts], dim=-1) if i < 4 else parts[0][i]
                    for i in range(6))
            else:
                q, k, v, raw_beta, f_a, g_a = self.in_proj_qkvbfg_a(x).split(sizes, dim=-1)
        qkv, histories = [], []
        for projected, weight, history in zip((q,k,v),
                                        (self.q_conv,self.k_conv,self.v_conv),state.conv,strict=True):
            conv = causal_conv if self.backend=='torch' else ascendc_causal_conv
            value, tail = conv(projected,weight.squeeze(1),history)
            qkv.append(value.reshape(-1,self.num_heads,self.head_dim)); histories.append(tail)
        raw_gate = self.f_b_proj(f_a).reshape(-1,self.num_heads,self.head_dim)
        if type(self.chunk_prefill_single) is not bool or (self.chunk_prefill_single and self.backend != 'ascendc_chunk_prefill'):
            raise ValueError('singleton prefill requires explicit chunk backend')
        if self.backend == 'ascendc_chunk_prefill' and (x.shape[0] > 1 or self.chunk_prefill_single):
            from .chunk_prefill import chunk_prefill_sharded
            output, recurrent = chunk_prefill_sharded(*qkv,raw_gate,raw_beta,state.recurrent,self.a_log,self.dt_bias,shards=4,lower_bound=self.lower_bound)
        else:
            core = {'torch':recurrent_kda,'ascendc':ascendc_recurrent_kda,'ascendc_installed_decode':installed_decode_recurrent_kda,'ascendc_chunk_prefill':ascendc_recurrent_kda}[self.backend]
            output, recurrent = core(*qkv,raw_gate,raw_beta,state.recurrent,self.a_log,self.dt_bias,lower_bound=self.lower_bound)
        gate = self.g_b_proj(g_a).reshape(-1,self.num_heads,self.head_dim)
        normed = output.float() * torch.rsqrt(output.float().square().mean(-1,keepdim=True)+self.norm_eps)
        gated = (normed*self.o_norm_weight.float()*gate.float().sigmoid()).to(x.dtype)
        result = self.o_proj(gated.reshape(x.shape[0],self.num_heads*self.head_dim))
        return result, KDAState(tuple(histories),recurrent,state.tokens_seen+x.shape[0])
