# SPDX-License-Identifier: Apache-2.0
"""Bounded expert-owner route partition and exact-order merge preparation.

No remote dispatch, collectives, model registration or NPU execution is
provided. Each rank's outputs must be returned in its partition row order.
Reassembly restores global packed order before the existing weighted combine,
so the unscaled shared expert is added exactly once after all routed outputs.
"""
from dataclasses import dataclass
import torch
from .moe import PackedRoutes,combine_packed
from .expert_bank import compact_expert_plan

@dataclass(frozen=True)
class RankPartition:
    rank: int
    packed_indices: torch.Tensor
    routes: PackedRoutes
    active_global_expert_ids: torch.Tensor

@dataclass(frozen=True)
class RankRoutePlan:
    routes: PackedRoutes
    partitions: tuple[RankPartition,...]
    world_size: int


def partition_routes(routes, *, world_size, max_packed_rows):
    if type(world_size) is not int or world_size not in (1,4):
        raise ValueError('world size must be explicitly 1 or 4')
    if type(max_packed_rows) is not int or max_packed_rows < 0:
        raise ValueError('nonnegative packed-row bound required')
    if not isinstance(routes,PackedRoutes):
        raise ValueError('validated packed routes required')
    compact_expert_plan(routes)
    if routes.expert_ids.numel()>max_packed_rows:
        raise ValueError('packed routes exceed explicit bound')
    parts=[]
    for rank in range(world_size):
        indices=(routes.expert_ids.remainder(world_size)==rank).nonzero().flatten()
        ids=routes.expert_ids.index_select(0,indices)
        tokens=routes.token_indices.index_select(0,indices)
        weights=routes.weights.index_select(0,indices)
        ends=torch.bincount(ids,minlength=routes.num_experts).cumsum(0)
        part=PackedRoutes(tokens,ids,weights,ends,routes.num_tokens,routes.num_experts)
        compact=compact_expert_plan(part)
        parts.append(RankPartition(rank,indices,part,compact.global_expert_ids))
    return RankRoutePlan(routes,tuple(parts),world_size)


def combine_rank_outputs(plan, rank_outputs, *, shared_output=None):
    if not isinstance(plan,RankRoutePlan) or type(plan.world_size) is not int or plan.world_size not in (1,4):
        raise ValueError('explicit rank route plan required')
    routes=plan.routes
    compact_expert_plan(routes)
    if not isinstance(rank_outputs,(tuple,list)) or len(rank_outputs)!=plan.world_size or len(plan.partitions)!=plan.world_size:
        raise ValueError('exactly one output per rank required')
    # Recreate expected partitions to reject stale or mutated owner metadata.
    expected=partition_routes(routes,world_size=plan.world_size,max_packed_rows=routes.expert_ids.numel())
    result=None; hidden=None; dtype=None
    for rank,(part,want,out) in enumerate(zip(plan.partitions,expected.partitions,rank_outputs,strict=True)):
        if part.rank!=rank or part.routes.num_tokens!=want.routes.num_tokens or part.routes.num_experts!=want.routes.num_experts or not torch.equal(part.packed_indices,want.packed_indices) or not torch.equal(part.routes.expert_ids,want.routes.expert_ids) or not torch.equal(part.routes.token_indices,want.routes.token_indices) or not torch.equal(part.routes.weights,want.routes.weights) or not torch.equal(part.routes.group_ends,want.routes.group_ends) or not torch.equal(part.active_global_expert_ids,want.active_global_expert_ids):
            raise ValueError('rank partition is stale or inconsistent')
        if not isinstance(out,torch.Tensor) or out.ndim!=2 or out.shape[0]!=want.packed_indices.numel() or out.device!=routes.expert_ids.device or out.dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('rank output rows, device or dtype disagree')
        if result is None:
            hidden=out.shape[1];dtype=out.dtype
            result=torch.empty((routes.expert_ids.numel(),hidden),dtype=dtype,device=out.device)
        elif out.shape[1]!=hidden or out.dtype!=dtype:
            raise ValueError('rank output geometry or dtype differs')
        result.index_copy_(0,want.packed_indices,out)
    return combine_packed(result,routes,shared_output=shared_output)
