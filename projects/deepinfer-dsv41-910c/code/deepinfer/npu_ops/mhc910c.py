# SPDX-License-Identifier: Apache-2.0
# Mathematical contracts follow TileKernels 2.1.0 tile_kernels/torch/mhc.py.
"""Portable eager implementations of DeepInfer's mHC operator interfaces."""
import torch


def default_n_splits():return 1

def expand_to_mhc(hidden,mhc_mult):
    if mhc_mult<1:raise ValueError('mhc_mult must be positive')
    return hidden.unsqueeze(-2).expand(*hidden.shape[:-1],mhc_mult,hidden.shape[-1]).contiguous()

def sinkhorn_normalize(x,repeat=10,eps=1e-6):
    if repeat<1:raise ValueError('Sinkhorn requires at least one iteration')
    out=x.float().softmax(-1)+eps
    out=out/(out.sum(-2,keepdim=True)+eps)
    for _ in range(repeat-1):
        out=out/(out.sum(-1,keepdim=True)+eps)
        out=out/(out.sum(-2,keepdim=True)+eps)
    return out.to(x.dtype)

def mhc_head_compute_mix(input_mix,mhc_scale,mhc_base,mhc_pre_eps):
    return torch.sigmoid(input_mix*mhc_scale+mhc_base)+mhc_pre_eps

def mhc_pre_apply_mix(x,mix):return (x.float()*mix).sum(-2).bfloat16()

def mhc_post(x,residual,post_layer_mix,comb_res_mix):
    mixed=torch.matmul(comb_res_mix.transpose(-2,-1),residual.float())
    return (x.float().unsqueeze(-2)*post_layer_mix+mixed).bfloat16()

def mhc_pre_split_mixes(input_mixes,mhc_scale,mhc_base,mhc_mult,mhc_post_mult_value,mhc_pre_eps):
    m=mhc_mult
    if input_mixes.shape[-1]!=m*(m+2) or mhc_scale.shape!=(3,) or mhc_base.shape!=(m*(m+2),):raise ValueError('invalid mHC mix shapes')
    pre=torch.sigmoid(input_mixes[...,:m]*mhc_scale[0]+mhc_base[:m])[...,None]+mhc_pre_eps
    post=(torch.sigmoid(input_mixes[...,m:2*m]*mhc_scale[1]+mhc_base[m:2*m])*mhc_post_mult_value)[...,None]
    combination=(input_mixes[...,2*m:]*mhc_scale[2]+mhc_base[2*m:]).reshape(*input_mixes.shape[:-1],m,m)
    return pre,post,combination

def mhc_prenorm_gemm(x,fn,n_splits):
    channels,k=fn.shape
    if x.shape[-2]*x.shape[-1]!=k or n_splits<1 or k%n_splits:raise ValueError('invalid prenorm dimensions or split count')
    flat=x.reshape(-1,k).float();step=k//n_splits;products=[];squares=[]
    for split in range(n_splits):
        part=flat[:,split*step:(split+1)*step]
        products.append(torch.mm(part,fn[:,split*step:(split+1)*step].float().T)[:,None,:]);squares.append(part.square().sum(-1,keepdim=True))
    return torch.stack(products),torch.stack(squares)

def mhc_pre_norm_fn(residual,mhc_fn,mhc_norm_weight,mhc_norm_eps,n_splits=None):
    fn=mhc_fn if mhc_norm_weight is None else mhc_fn*mhc_norm_weight
    products,squares=mhc_prenorm_gemm(residual,fn,n_splits or 1)
    out=products.sum(0)[:,0,:]*torch.rsqrt(squares.sum(0)/fn.shape[1]+mhc_norm_eps)
    return out.reshape(*residual.shape[:-2],fn.shape[0])

def mhc_pre_big_fuse(residual,fn,mhc_scale,mhc_base,rms_eps,mhc_pre_eps,mhc_sinkhorn_eps,mhc_post_mult_value,sinkhorn_repeat,n_splits=None):
    mixes=mhc_pre_norm_fn(residual,fn,None,rms_eps,n_splits)
    pre,post,comb=mhc_pre_split_mixes(mixes,mhc_scale,mhc_base,residual.shape[-2],mhc_post_mult_value,mhc_pre_eps)
    comb=sinkhorn_normalize(comb,sinkhorn_repeat,mhc_sinkhorn_eps)
    return post,comb,mhc_pre_apply_mix(residual,pre)
