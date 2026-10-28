"""Correctness-first routed MoE primitives for the 910C reference profile.

Dispatch/combine uses ordinary HCCL all-to-all, contiguous expert ownership,
and an explicit local-expert callback. This is not a drop-in DeepEP buffer
implementation: EPLB, asynchronous overlap and graph capture are not supported.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F


def route(logits, bias, topk, scale=1., *, image_bias=None, image_mask=None, valid_mask=None):
    if logits.ndim!=2 or not 0<topk<=logits.shape[1]:raise ValueError('invalid routing dimensions')
    if bias.shape!=(logits.shape[1],) or bias.dtype!=torch.float32:raise ValueError('routing bias must be FP32 [experts]')
    scores=F.softplus(logits.float()).sqrt()
    selection=scores+bias
    if image_bias is not None:
        if image_mask is None or image_bias.shape!=bias.shape or image_mask.shape!=logits.shape[:1]:raise ValueError('invalid image routing bias or mask')
        selection=scores+torch.where(image_mask[:,None],image_bias[None],bias[None])
    indices=torch.topk(selection,topk,dim=-1,sorted=True).indices
    weights=scores.gather(-1,indices)
    weights=weights/(weights.sum(-1,keepdim=True)+1e-20)*scale
    if valid_mask is not None:
        if valid_mask.shape!=logits.shape[:1] or valid_mask.dtype!=torch.bool:raise ValueError('invalid token mask')
        indices=torch.where(valid_mask[:,None],indices,-1);weights=torch.where(valid_mask[:,None],weights,0.)
    return indices.to(torch.int64),weights


def swiglu(gate,up,weights=None,clamp=None):
    gate,up=gate.float(),up.float()
    if clamp is not None:gate=gate.clamp_max(clamp);up=up.clamp(-clamp,clamp)
    activated=F.silu(gate)*up
    if weights is not None:activated=activated*weights[:,None]
    return activated.to(torch.bfloat16)


def dispatch_combine(hidden:torch.Tensor,indices:torch.Tensor,weights:torch.Tensor,num_experts:int,local_expert:Callable,group=None,*,weight_on_output=False,local_batch=None,trusted_indices=False):
    """Execute selected local/remote experts and return this rank's token output.

    ``local_expert(global_id, BF16_inputs, FP32_route_weights)`` applies routing
    weights once, before down projection, and returns BF16 outputs. With
    ``weight_on_output=True`` the callback returns unweighted expert outputs;
    route weights are applied in FP32 after return before combining, as used
    by the Ascend W8A8 path. Negative
    indices mark invalid routes. No silent FP8 cast or topology emulation.
    """
    if hidden.ndim!=2 or hidden.dtype!=torch.bfloat16 or indices.ndim!=2 or indices.shape!=weights.shape or indices.shape[0]!=hidden.shape[0]:raise ValueError('invalid MoE tensor shapes or BF16 input')
    if indices.dtype!=torch.int64 or weights.dtype!=torch.float32 or hidden.device!=indices.device or hidden.device!=weights.device:raise ValueError('MoE indices/weights must be int64/FP32 on the input device')
    if num_experts<=0 or (not trusted_indices and bool(((indices < -1)|(indices>=num_experts)).any())):raise ValueError('invalid expert IDs')
    initialized=dist.is_initialized();world=dist.get_world_size(group) if initialized else 1;rank=dist.get_rank(group) if initialized else 0
    if num_experts%world:raise ValueError('expert count must divide evenly across the group')
    local_count=num_experts//world;m,hidden_dim=hidden.shape;k=indices.shape[1]
    flat=indices.flatten();selected=torch.nonzero(flat>=0,as_tuple=False).flatten()
    expert_ids=flat[selected];owners=expert_ids//local_count
    # Owner IDs are exactly representable in FP32 at this world size. Ascend
    # ArgSort on INT64 falls back to AICPU; FP32 keeps dispatch sorting on-core.
    order=torch.argsort(owners.float() if world<2**24 else owners)
    selected=selected[order];expert_ids=expert_ids[order];owners=owners[order]
    inputs=hidden.index_select(0,selected//k).contiguous();route_weights=weights.flatten()[selected].contiguous()
    send_counts=torch.bincount(owners,minlength=world).cpu().tolist()
    if world>1:
        count_tensor=torch.tensor(send_counts,device=hidden.device,dtype=torch.int64);received_counts=torch.empty_like(count_tensor)
        dist.all_to_all_single(received_counts,count_tensor,group=group)
        recv_counts=received_counts.cpu().tolist()
        global_routes=torch.tensor([selected.numel()],device=hidden.device,dtype=torch.int64);dist.all_reduce(global_routes,group=group)
        if global_routes.item()==0:return torch.zeros_like(hidden)
        def exchange(tensor):
            output=torch.empty((sum(recv_counts),*tensor.shape[1:]),dtype=tensor.dtype,device=tensor.device)
            dist.all_to_all_single(output,tensor,output_split_sizes=recv_counts,input_split_sizes=send_counts,group=group)
            return output
        recv_x=exchange(inputs);recv_ids=exchange(expert_ids.contiguous());recv_weights=exchange(route_weights)
    else:
        recv_x,recv_ids,recv_weights=inputs,expert_ids,route_weights;recv_counts=send_counts
    if local_batch is not None:
        values=local_batch(recv_x,recv_ids-rank*local_count)
    else:
        values=torch.empty_like(recv_x)
        for expert in torch.unique(recv_ids).cpu().tolist():
            if not rank*local_count<=expert<(rank+1)*local_count:raise RuntimeError('received a non-local expert')
            positions=torch.nonzero(recv_ids==expert,as_tuple=False).flatten()
            result=local_expert(expert,recv_x.index_select(0,positions),recv_weights.index_select(0,positions))
            if result.shape!=(positions.numel(),hidden_dim) or result.dtype!=hidden.dtype or result.device!=hidden.device:raise ValueError('local expert returned incompatible output')
            values.index_copy_(0,positions,result)
    if world>1:
        returned=torch.empty_like(inputs)
        dist.all_to_all_single(returned,values,output_split_sizes=send_counts,input_split_sizes=recv_counts,group=group)
    else:returned=values
    combined=torch.zeros((m,hidden_dim),device=hidden.device,dtype=torch.float32)
    values_to_combine=returned.float()
    if weight_on_output:values_to_combine=values_to_combine*route_weights[:,None]
    combined.index_add_(0,selected//k,values_to_combine)
    return combined.to(hidden.dtype)
