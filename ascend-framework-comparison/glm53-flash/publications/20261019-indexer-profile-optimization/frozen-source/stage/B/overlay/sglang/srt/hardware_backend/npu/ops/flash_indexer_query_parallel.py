# SPDX-License-Identifier: Apache-2.0
"""Partition independent prefill query rows inside the native attention TP group.

Cache compression/state ownership stay unchanged. Each rank scores its rows;
the final native int32 token indices are gathered back in original row order.
"""
import torch
from sglang.srt.hardware_backend.npu.ops.flash_kpool_score import glm5_next_lightning_indexer_triton

def _local_audit_context(context,start,count):
    if context is None:return None
    value=dict(context);lengths=[];starts=[];offset=0
    for length,sequence in zip(context['query_lengths'],context['sequences']):
        left=min(max(start-offset,0),length)
        right=min(max(start+count-offset,0),length)
        lengths.append(max(0,right-left));starts.append(sequence-length+left)
        offset+=length
    value.update(query_lengths=lengths,query_start_positions=starts)
    return value

def query_parallel_indexer(query,cache,weights,query_ends,pool_lens,block_table,positions,*,group,rank,world_size,prefill,**kwargs):
    if type(world_size) is not int or world_size<1 or not 0<=rank<world_size:raise ValueError('Invalid actual attention TP rank')
    total=query.shape[0]
    if not prefill or world_size==1 or total<world_size*32:
        return glm5_next_lightning_indexer_triton(query,cache,weights,query_ends,pool_lens,block_table,positions,**kwargs)
    if kwargs.get('output_buffer') is not None:raise ValueError('Query-parallel prefill does not replace an existing output buffer')
    chunk=(total+world_size-1)//world_size;start=min(rank*chunk,total);count=min(chunk,total-start)
    width=kwargs['index_topk']+kwargs['index_kpool']-1
    # All ranks enter exactly one same-shape collective, including empty tails.
    padded=torch.full((chunk,width),-1,dtype=torch.int32,device=query.device)
    if count:
        local_ends=torch.clamp(query_ends-start,min=0,max=count)
        local_kwargs=dict(kwargs)
        local_kwargs['audit_context']=_local_audit_context(kwargs.get('audit_context'),start,count)
        local=glm5_next_lightning_indexer_triton(query[start:start+count],cache,weights[start:start+count],local_ends,pool_lens,block_table,positions[start:start+count],**local_kwargs)
        padded[:count].copy_(local.squeeze(1))
    gathered=group.all_gather(padded,dim=0)
    if gathered.dtype!=torch.int32 or gathered.shape!=(chunk*world_size,width):raise ValueError('Native attention-TP gather returned wrong token-index ABI')
    return gathered[:total].unsqueeze(1)
