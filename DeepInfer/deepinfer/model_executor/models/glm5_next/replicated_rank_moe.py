# SPDX-License-Identifier: Apache-2.0
"""Explicit replicated-input, all-gather expert correctness path.

Each rank loads only its modulo-owned routed experts; rank zero loads shared.
Outputs are gathered in fixed bounded buffers and restored to global packed
order before weighting and adding the shared output exactly once. This is a
correctness path, not all-to-all dispatch, a full TP model or serving code.
"""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from .moe import pack_routes, dispatch_hidden
from .expert_bank import compact_expert_plan
from .rank_routes import partition_routes,combine_rank_outputs
from .cached_expert_bank import cached_selected_expert_bank
from .cached_expert_session import CachedExpertBankSession, session_selected_expert_bank
from .coordinated_bank_lease import coordinated_bank_lease, coordinated_local_stage

@torch.inference_mode()
def replicated_rank_moe(reader, hidden, routes, *, layer_id, rank, world_size,
                        cache_bytes, max_assembled_bytes, max_packed_rows,
                        max_checkpoint_bytes=512*1024**2,
                        max_resident_bytes=512*1024**2, group=None, cache_session=None):
    if type(world_size) is not int or world_size not in (1,2,4,8,16) or type(rank) is not int or not 0<=rank<world_size:
        raise ValueError('explicit rank and supported world required')
    if world_size>1:
        if not dist.is_initialized() or dist.get_rank(group)!=rank or dist.get_world_size(group)!=world_size:
            raise ValueError('matching initialized group required')
    elif dist.is_initialized():
        raise ValueError('world-one path requires no initialized group')

    # A local route error or unequal tensor shape must not send peers into
    # different-sized collectives.  Each live rank first exchanges the same
    # fixed-size metadata/status vector on its current NPU. This does not
    # cover process death, device faults, or errors after expert loading.
    local_error=None
    try:
        if hidden.device.type!='npu' or hidden.dtype!=torch.bfloat16 or hidden.ndim!=2 or hidden.shape[0]!=1:
            raise ValueError('one BF16 NPU token required')
        if cache_session is not None:
            if not isinstance(cache_session,CachedExpertBankSession):
                raise ValueError('explicit reusable expert bank session required')
            try:cache_session.validate_binding(reader,rank=rank,world_size=world_size,cache_bytes=cache_bytes,max_assembled_bytes=max_assembled_bytes)
            except RuntimeError as exc:raise ValueError('cache session is closed') from exc
        packed=pack_routes(routes,num_experts=reader.config['text_config']['n_routed_experts'])
        if packed.num_tokens!=hidden.shape[0] or packed.expert_ids.device!=hidden.device:
            raise ValueError('routing and token inputs must share device and count')
        plan=partition_routes(packed,world_size=world_size,max_packed_rows=max_packed_rows)
    # Unexpected synchronous route/session preparation errors must also
    # participate in the preflight, provided the group and device remain usable.
    except Exception as exc:
        local_error=exc
    if world_size>1:
        if local_error is None:
            # Fixed-length vector: validity, tensor geometry/dtypes and
            # layer/packing parameters that determine collective shapes.
            codes={torch.bfloat16:1,torch.float32:2,torch.int32:3,torch.int64:4}
            meta=[0,hidden.shape[0],hidden.shape[1],codes.get(hidden.dtype,0),
                  routes.expert_ids.shape[0],routes.expert_ids.shape[1],codes.get(routes.expert_ids.dtype,0),
                  routes.weights.shape[0],routes.weights.shape[1],codes.get(routes.weights.dtype,0),
                  layer_id,max_packed_rows,packed.num_experts,int(cache_session is not None)]
        else:
            meta=[1]+[0]*13
        value=torch.tensor(meta,dtype=torch.int64,device='npu')
        copies=[torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(copies,value,group=group)
        if any(int(x[0].item())!=0 for x in copies) or not all(torch.equal(copies[0],x) for x in copies[1:]):
            raise ValueError('rank input preflight failed or metadata differs') from local_error
    elif local_error is not None:
        raise local_error
    if world_size>1:
        # Compare actual replicated tensors before loading any weights. Every
        # rank performs the same collectives and checks the same inputs.
        for value in (hidden.contiguous(),routes.expert_ids.contiguous(),routes.weights.contiguous()):
            copies=[torch.empty_like(value) for _ in range(world_size)]
            dist.all_gather(copies,value,group=group)
            if not all(torch.equal(copies[0],other) for other in copies[1:]):
                raise ValueError('ranks require identical token and routing inputs')
    # Defer partition extraction, device-to-host conversion and lease factory
    # until coordinated_bank_lease enters the context. A local preparation
    # exception must not leave peers waiting in its status collective.
    @contextmanager
    def prepare_bank():
        part=plan.partitions[rank]
        ids=part.active_global_expert_ids.detach().cpu().tolist()
        options=dict(rank=rank,world_size=world_size,cache_bytes=cache_bytes,
                     max_assembled_bytes=max_assembled_bytes,max_experts=max_packed_rows,
                     include_shared=(rank==0),max_checkpoint_bytes=max_checkpoint_bytes,
                     max_resident_bytes=max_resident_bytes)
        lease=(cached_selected_expert_bank(reader,layer_id,ids,hidden.device,**options)
               if cache_session is None else
               session_selected_expert_bank(cache_session,reader,layer_id,ids,hidden.device,**options))
        with lease as (bank,cache):
            yield part,ids,bank,cache
    with coordinated_bank_lease(prepare_bank(),device=hidden.device,rank=rank,world_size=world_size,group=group) as (part,ids,bank,cache):
        def local_stage():
            if ids:
                compact=compact_expert_plan(part.routes,bank.global_expert_ids)
                local=bank.bank.forward_grouped(dispatch_hidden(hidden,part.routes),compact.group_ends,group_list_type=0)
            else:
                local=hidden.new_empty((0,hidden.shape[1]))
            if local.shape!=(part.packed_indices.numel(),hidden.shape[1]) or local.dtype!=hidden.dtype:
                raise ValueError('local expert output geometry differs')
            shared=bank.shared(hidden) if rank==0 else torch.empty_like(hidden)
            # Prepare every rank-local exchange buffer before coordinating the
            # stage: an allocation, copy or statistics error must be reported
            # before any peer enters broadcast or all_gather.
            stats={k:v for k,v in cache.stats().items() if k!='keys_lru'}
            padded=gathered=None
            if world_size>1:
                capacity=max(1,packed.expert_ids.numel())
                padded=hidden.new_zeros((capacity,hidden.shape[1]));padded[:len(local)].copy_(local)
                gathered=[torch.empty_like(padded) for _ in range(world_size)]
            # A broken device/collective cannot be recovered here.
            import torch_npu
            torch_npu.npu.synchronize()
            return local,shared,stats,padded,gathered
        local,shared,stats,padded,gathered=coordinated_local_stage(local_stage,device=hidden.device,rank=rank,world_size=world_size,group=group)
        if world_size>1:
            dist.broadcast(shared,src=(dist.get_global_rank(group,0) if group is not None else 0),group=group)
            dist.all_gather(gathered,padded,group=group)
            outputs=[value[:plan.partitions[i].packed_indices.numel()] for i,value in enumerate(gathered)]
        else:
            outputs=[local]
        # A rank-local combine or synchronous device error must be agreed
        # before peers can leave their leases and enter the next layer.
        def combine_stage():
            result=combine_rank_outputs(plan,outputs,shared_output=shared)
            import torch_npu
            torch_npu.npu.synchronize()
            return result
        combined=coordinated_local_stage(combine_stage,device=hidden.device,rank=rank,world_size=world_size,group=group)
        del bank,local,shared,outputs
    # Every rank must observe a post-lease stats failure before advancing.
    def final_cache_stats():
        final=cache.stats()
        return final['closed'],final['used_bytes']
    stats['closed_after_context'],stats['used_after_context']=coordinated_local_stage(
        final_cache_stats,device=hidden.device,rank=rank,world_size=world_size,group=group)
    return combined,{'rank':rank,'world_size':world_size,'local_ids':ids,'local_rows':part.packed_indices.numel(),
                     'shared_rank':0,'cache':stats}
