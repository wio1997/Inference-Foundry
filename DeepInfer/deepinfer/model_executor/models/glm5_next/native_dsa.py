# SPDX-License-Identifier: Apache-2.0
"""Unsharded native GLM NoPE DSA for one request and <=128 total tokens.

All causal tokens fit below topk2048, so no pool ranking is needed. Cache
state uses the existing DSAState append; SFA sees a private one-page copy. The
current tested CANN binary rejects absent RoPE, so explicit all-zero 64-wide
RoPE tensors preserve NoPE scores for this bounded path. No native page leases,
TP/EP, long-context pool selection, MTP, graphs or throughput claim.
"""
import torch
from .cache import append_dsa


class NativeShortDSA:
    def __init__(self,operator,*,max_total_tokens=128):
        if type(max_total_tokens) is not int or not 1<=max_total_tokens<=256:raise ValueError('explicit context bound must be 1..256')
        self.max_total_tokens=max_total_tokens
        if not callable(operator):raise ValueError('explicit native sparse flash attention operator required')
        self.operator=operator;self.mla=None;self.indexer=None

    def bind(self,mla,indexer):
        if self.mla is not None:raise ValueError('native short DSA operation already bound')
        if (mla.num_heads,mla.qk_dim,mla.kv_rank,mla.value_dim)!=(64,256,512,256):raise ValueError('GLM MLA geometry differs')
        if (indexer.layout.latent_dim,indexer.layout.index_dim,indexer.layout.pool_size,indexer.layout.topk_tokens,indexer.layout.logical_block_size,indexer.layout.lookahead)!=(512,128,4,2048,128,0):raise ValueError('short DSA cache layout differs')
        weight=getattr(mla.kv_b,'weight',None)
        if not isinstance(weight,torch.Tensor) or weight.shape!=(64*512,512) or weight.dtype!=torch.bfloat16:raise ValueError('FLOAT BF16 kv_b NK weight required')
        self.mla,self.indexer=mla,indexer
        return self

    @torch.inference_mode()
    def __call__(self,step,state):
        if self.mla is None or self.indexer is None:raise ValueError('native short DSA must bind checkpoint projections')
        if step.mla.query.device.type!='npu':raise ValueError('native short DSA requires NPU')
        count=len(step.positions);start=state.tokens_seen;end=start+count
        if not 1<=count<=128 or end>self.max_total_tokens:raise ValueError('native DSA exceeds explicit context bound')
        if step.scale!=1/16 or not torch.equal(step.positions.cpu(),torch.arange(start,end,dtype=torch.int64)):raise ValueError('native DSA positions/scale differ')
        if step.mla.query.shape!=(count,64,256) or step.mla.latent.shape!=(count,512):raise ValueError('native DSA projection geometry differs')
        next_state=append_dsa(step.mla.latent,step.indexer.raw_keys,step.indexer.gate_scores,self.indexer.ape,state,self.indexer.layout,start_position=start)
        weight=self.mla.kv_b.weight.reshape(64,512,512)
        key=weight[:,:256,:].contiguous();value=weight[:,256:,:].transpose(1,2).contiguous()
        query=torch.bmm(step.mla.query.transpose(0,1).contiguous(),key).transpose(0,1).contiguous()
        pages=(end+127)//128;cache=torch.zeros((pages,128,1,512),device=query.device,dtype=torch.bfloat16);cache.view(-1,512)[:end]=next_state.latent
        indices=torch.full((count,1,2051),-1,dtype=torch.int32)
        for row in range(count):indices[row,0,:start+row+1]=torch.arange(start+row+1,dtype=torch.int32)
        to_device=lambda tensor:tensor.to(query.device)
        result=self.operator(query=query,key=cache,value=cache,sparse_indices=to_device(indices),scale_value=step.scale,sparse_block_size=1,
            block_table=to_device(torch.arange(pages,dtype=torch.int32).reshape(1,-1)),actual_seq_lengths_query=to_device(torch.tensor([count],dtype=torch.int32)),
            actual_seq_lengths_kv=to_device(torch.tensor([end],dtype=torch.int32)),
            query_rope=torch.zeros((count,64,64),device=query.device,dtype=torch.bfloat16),
            key_rope=torch.zeros((pages,128,1,64),device=query.device,dtype=torch.bfloat16),
            layout_query='TND',layout_kv='PA_BSND',sparse_mode=3,attention_mode=2,return_softmax_lse=False)[0]
        if result.shape!=(count,64,512) or result.dtype!=torch.bfloat16:raise ValueError('native SFA latent output differs')
        values=torch.bmm(result.transpose(0,1).contiguous(),value).transpose(0,1).contiguous()
        if not bool(torch.isfinite(values).all()):raise ValueError('native SFA output must be finite')
        # Native execution is synchronized before publishing the new state.
        import torch_npu
        torch_npu.npu.synchronize()
        return values,next_state


def make_dsa_operations(config,*,sfa_library,max_total_tokens=128):
    """Load one explicitly supplied binary and bind each configured DSA layer."""
    if sfa_library is None:raise ValueError('explicit compatible SFA library path required')
    torch.ops.load_library(str(sfa_library))
    operator=torch.ops._C_ascend.npu_sparse_flash_attention
    return {s.layer_id:NativeShortDSA(operator,max_total_tokens=max_total_tokens) for s in config.layers if s.attention=='deepseek_sparse_attention'}
