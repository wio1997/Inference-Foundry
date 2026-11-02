# SPDX-License-Identifier: Apache-2.0
"""Unfused GLM mHC equations, with explicit residual-stream orientation.

Matches the vLLM torch reference: flattened RMS for gate logits, positive
pre coefficients, scaled sigmoid post coefficients and alternating Sinkhorn
row/column normalizations. Parameters stored as BF16 in the checkpoint become
FP32 runtime gate parameters. These are correctness components; native fused
mHC operator compatibility and throughput are separate validation work.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class HCMix:
    post: torch.Tensor  # [tokens, output_stream, 1], FP32
    comb: torch.Tensor  # [tokens, input_stream, output_stream], FP32


def rms_norm(x, weight, eps):
    if weight.shape != (x.shape[-1],) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('invalid RMSNorm weight or epsilon')
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1,keepdim=True)+eps) * weight.float()).to(x.dtype)


class GlmMHC(nn.Module):
    def __init__(self, fn, base, scale, *, streams, hidden_size, rms_eps=1e-5,
                 hc_eps=1e-6, sinkhorn_iters=20, post_mult=2.0):
        super().__init__()
        if type(streams)!=int or streams<1 or type(hidden_size)!=int or hidden_size<1:
            raise ValueError('invalid mHC dimensions')
        width=streams*(streams+2)
        if fn.shape!=(width,streams*hidden_size) or base.shape!=(width,) or scale.shape!=(3,):
            raise ValueError('invalid mHC parameter layout')
        if type(sinkhorn_iters)!=int or sinkhorn_iters<1:
            raise ValueError('Sinkhorn requires at least one iteration')
        if not all(math.isfinite(v) and v>0 for v in (rms_eps,hc_eps,post_mult)):
            raise ValueError('mHC epsilons and post multiplier must be finite and positive')
        if len({t.device for t in (fn,base,scale)})!=1:
            raise ValueError('mHC parameters must share a device')
        self.streams,self.hidden_size=streams,hidden_size
        self.backend="torch"  # optional installed Ascend implementation is explicit
        self.rms_eps,self.hc_eps=rms_eps,hc_eps
        self.sinkhorn_iters,self.post_mult=sinkhorn_iters,post_mult
        for name,value in [('fn',fn),('base',base),('scale',scale)]:
            self.register_buffer(name,value.float())

    @classmethod
    def from_checkpoint(cls,reader,layer_id,kind,device='cpu'):
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        c=Glm5NextConfig.from_checkpoint(reader.root)
        if not 0<=layer_id<c.num_hidden_layers or kind not in ('attn','ffn'):
            raise ValueError('invalid GLM mHC backbone component')
        text=c.hf_config['text_config']
        if not c.mhc_config['mhc'] or text.get('hres_vwnstyle',True) is not True or text.get('mhc_no_norm_weight',False) not in (None,False):
            raise ValueError('unsupported GLM mHC variant')
        prefix=f'model.language_model.layers.{layer_id}.{kind}_hc'
        values=[]
        for suffix in ('fn','base','scale'):
            value,dtype=reader.tensor(prefix+'.'+suffix)
            if dtype not in ('BF16','F32'):
                raise ValueError('mHC parameters must have floating storage')
            values.append(value.to(device))
        return cls(*values,streams=c.mhc_config['hc_mult'],hidden_size=c.hidden_size,
                   rms_eps=text['rms_norm_eps'],hc_eps=c.mhc_config['hc_eps'],
                   sinkhorn_iters=c.mhc_config['hc_sinkhorn_iters'],
                   post_mult=text.get('mhc_post_mult_value',2.0))

    def expand(self,x):
        if x.ndim!=2 or x.shape[1]!=self.hidden_size:
            raise ValueError('expected collapsed hidden states [tokens,hidden]')
        return x.unsqueeze(1).expand(-1,self.streams,-1).clone()

    def _validate_residual(self,residual):
        if residual.ndim!=3 or residual.shape[1:]!=(self.streams,self.hidden_size):
            raise ValueError('expected residual streams [tokens,streams,hidden]')
        if residual.dtype not in (torch.bfloat16,torch.float32) or residual.device!=self.fn.device:
            raise ValueError('mHC requires BF16/F32 streams on the parameter device')

    def pre(self,residual):
        self._validate_residual(residual)
        if self.backend == 'ascend_installed':
            if residual.device.type != 'npu':
                raise ValueError('installed mHC requires NPU tensors')
            x, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
                residual, self.fn, self.scale, self.base, self.streams,
                self.sinkhorn_iters, self.rms_eps, self.hc_eps)
            return x, HCMix(post.unsqueeze(-1) * (self.post_mult / 2.0), comb)
        if self.backend != 'torch':
            raise ValueError('unknown mHC backend')
        tokens=residual.shape[0];n=self.streams
        flat=residual.reshape(tokens,n*self.hidden_size).float()
        logits=(flat@self.fn.T)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+self.rms_eps)
        pre=(logits[:,:n]*self.scale[0]+self.base[:n]).sigmoid()+self.hc_eps
        post=(logits[:,n:2*n]*self.scale[1]+self.base[n:2*n]).sigmoid()*self.post_mult
        comb=(logits[:,2*n:]*self.scale[2]+self.base[2*n:]).reshape(tokens,n,n)
        comb=comb.softmax(-1)+self.hc_eps
        comb=comb/(comb.sum(-2,keepdim=True)+self.hc_eps)
        for _ in range(self.sinkhorn_iters-1):
            comb=comb/(comb.sum(-1,keepdim=True)+self.hc_eps)
            comb=comb/(comb.sum(-2,keepdim=True)+self.hc_eps)
        collapsed=(pre.unsqueeze(-1)*residual.float()).sum(1).to(residual.dtype)
        return collapsed,HCMix(post.unsqueeze(-1),comb)

    def post(self,output,residual,mix):
        self._validate_residual(residual)
        tokens=residual.shape[0];n=self.streams
        if output.shape!=(tokens,self.hidden_size) or output.dtype!=residual.dtype or output.device!=residual.device:
            raise ValueError('mHC sublayer output does not match residual')
        if mix.post.shape!=(tokens,n,1) or mix.comb.shape!=(tokens,n,n):
            raise ValueError('invalid mHC mixing layout')
        if any(t.device!=residual.device or t.dtype!=torch.float32 for t in (mix.post,mix.comb)):
            raise ValueError('mHC mixing coefficients must be FP32 on residual device')
        if self.backend == 'ascend_installed':
            if residual.device.type != 'npu':
                raise ValueError('installed mHC requires NPU tensors')
            return torch.ops._C_ascend.npu_hc_post(
                output.unsqueeze(0), residual.unsqueeze(0),
                mix.post.squeeze(-1).unsqueeze(0), mix.comb.unsqueeze(0)).squeeze(0)
        if self.backend != 'torch':
            raise ValueError('unknown mHC backend')
        mixed=torch.bmm(mix.comb.transpose(1,2),residual.float())
        return (mixed+mix.post*output.float().unsqueeze(1)).to(residual.dtype)
