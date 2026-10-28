# SPDX-License-Identifier: Apache-2.0
"""Native MC2 dispatch/GMM/combine using the pinned Ascend A3 operator ABI.

The ABI is adapted from the image's Apache-2.0 TokenDispatcherWithMC2.
No vendor runtime/config/parallel-state Python modules are imported. DeepInfer
owns the HCCL group, INT8 expert bank, routing and token capacity.
"""
import torch
import torch.distributed as dist
import os

_MC2_GROUP=None


def initialize_mc2_group(buffer_mib=2560):
    """Use a dedicated MC2 communicator with the documented pg_options API.

    The A3 kernel reported 2,145,386,496 bytes required at global_bs=4096;
    the normal HCCL group's default200MiB window is insufficient. Restrict
    the larger window to MC2 instead of multiplying it across TP/control.
    """
    global _MC2_GROUP
    if _MC2_GROUP is None:
        import torch_npu
        options=torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config={'hccl_buffer_size':buffer_mib}
        _MC2_GROUP=dist.new_group(list(range(dist.get_world_size())),backend='hccl',pg_options=options)
        dist.barrier(group=_MC2_GROUP)
    return _MC2_GROUP


class NativeMC2:
    def __init__(self,group=None):
        if _MC2_GROUP is None:
            raise RuntimeError('initialize_mc2_group() before constructing the native MC2 model')
        self.group=_MC2_GROUP
        self.rank=dist.get_rank(self.group)
        self.world=dist.get_world_size(self.group)
        self.name=self.group._get_backend(torch.device('npu')).get_hccl_comm_name(self.rank)
        self.fused_calls=0
        self.fused_max_output_size=0
        self.capacity_floor=int(os.environ.get('DEEPINFER_910C_MC2_CAPACITY_FLOOR','65536'))
        if not 256<=self.capacity_floor<=65536 or self.capacity_floor%256:
            raise ValueError('MC2 capacity floor must be a multiple of256 in256..65536')
        self.capacity_records={}
        self._fused_masks={}
        # Ignored by the INT8 C++ adapter, retained for its Tensor[] ABI.
        self._empty_bias=[torch.tensor([],dtype=torch.float32)]

    def __call__(self,x,ids,weights,bank,*,num_experts,global_bs,fused=False):
        if fused:
            return self._fused(x,ids,weights,bank,global_bs=global_bs)
        import torch_npu
        # The A3 profile passes the TP ABI fields even though experts are EP.
        common=dict(group_ep=self.name,ep_world_size=self.world,ep_rank_id=self.rank,
            expert_shard_type=0,shared_expert_rank_num=0,moe_expert_num=num_experts,
            global_bs=global_bs,group_tp=self.name,tp_world_size=1,tp_rank_id=0,comm_alg='')
        outputs=torch_npu.npu_moe_distribute_dispatch_v2(x=x,expert_ids=ids.int(),
            scales=None,quant_mode=0,expert_token_nums_type=0,**common)
        expanded,_,assist,groups,ep_counts,tp_counts,expand_scales=outputs[:7]
        values=bank.forward_grouped(expanded,groups,group_list_type=0)
        return torch_npu.npu_moe_distribute_combine_v2(expand_x=values,expert_ids=ids.int(),
            expert_scales=weights.float(),ep_send_counts=ep_counts,tp_send_counts=tp_counts,
            assist_info_for_combine=assist,expand_scales=expand_scales,comm_quant_mode=0,**common)

    def _fused(self,x,ids,weights,bank,*,global_bs):
        """Fused routed FFN, independent of vLLM's scheduler/parallel state.

        Unlike dispatch_v2's global_bs mode, this kernel sizes peer windows
        from local M. All ranks must therefore submit the same padded M;
        x_active_mask excludes padding from routing and expert computation.
        The prefill runner already provides a common global capacity.
        """
        capacity=global_bs//self.world
        count=x.shape[0]
        # Every route from every rank could target this rank's experts.
        # The floor defaults to the pinned65536 baseline. A smaller explicit
        # floor still retains the complete global route bound, even if all
        # active routes target this rank. Thus this never relies on observed
        # average routing or expert balance to avoid truncation.
        max_output_size=max(self.capacity_floor,global_bs*ids.shape[1])
        if (global_bs%self.world or not 0<count<=capacity
                or max_output_size>196608):
            raise ValueError('fused MC2 capacity must cover every rank and fit at most196608 routes')
        self.capacity_records[(global_bs,count,ids.shape[1])]=max_output_size
        if not hasattr(bank,'fused_first_scale'):
            raise RuntimeError('prepare_fused_mc2() before fused prefill')
        mask=self._fused_masks.get((capacity,count))
        if mask is None:
            mask=torch.arange(capacity,device=x.device)<count
            self._fused_masks[(capacity,count)]=mask
        if count<capacity:
            x=torch.nn.functional.pad(x,(0,0,0,capacity-count))
            ids=torch.nn.functional.pad(ids,(0,0,0,capacity-count))
            weights=torch.nn.functional.pad(weights,(0,0,0,capacity-count))
        out=torch.empty_like(x)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,weight1=[bank.first],weight2=[bank.second],expert_idx=ids.int(),
            scale1=[bank.fused_first_scale],scale2=[bank.fused_second_scale],
            bias1=self._empty_bias,bias2=self._empty_bias,probs=weights.float(),
            group=self.name,max_output_size=max_output_size,out=out,
            expert_token_nums=bank.fused_expert_tokens,x_active_mask=mask,
            swiglu_limit=0. if bank.limit is None else bank.limit,world_size=self.world)
        self.fused_calls+=1
        self.fused_max_output_size=max(self.fused_max_output_size,max_output_size)
        return out[:count]
