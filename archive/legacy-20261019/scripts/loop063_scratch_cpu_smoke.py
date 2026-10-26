#!/usr/bin/env python3
"""CPU mock of private metadata generation and commit, including SWA mapping."""
from types import SimpleNamespace
import torch
from runtime.target_metadata import CPGroupBinding, FixedTargetMetadataUpdater, RotaryBinding, SparseMetadataOperators

query=torch.arange(0,97,8,dtype=torch.int32)
full=torch.arange(1024*16,dtype=torch.float32).view(1024,1,1,16)
zero=torch.zeros_like(full)
block=torch.arange(12*8,dtype=torch.int32).view(12,8)
def group(ratio,swa=False):
 return CPGroupBinding(ratio=ratio,seq_lens=torch.zeros(12,dtype=torch.int32),input_positions=torch.zeros(96,dtype=torch.int64),start_pos=torch.zeros(12,dtype=torch.int32),local_query_start_loc=torch.zeros(13,dtype=torch.int32),local_seq_lens=torch.zeros(12,dtype=torch.int32),sas_metadata=torch.zeros(1024,dtype=torch.int32),qli_metadata=torch.zeros(1024,dtype=torch.int32) if ratio==4 else None,swa_slot_mapping=torch.zeros(96,2,dtype=torch.int32) if swa else None,swa_block_table=block if swa else None,swa_block_size=32 if swa else None)
ops=SparseMetadataOperators(sas=lambda **kw: torch.arange(1024,dtype=torch.int32)+int(kw['cmp_ratio']),qli=lambda **kw:torch.arange(1024,dtype=torch.int32)+7,cu_seqlens_ori_kv=torch.zeros(1,dtype=torch.int32),cu_seqlens_cmp_kv=torch.zeros(1,dtype=torch.int32),seqused_q=torch.zeros(1,dtype=torch.int32),device_name='cpu',num_heads=8,head_dim=16,sliding_window=128,index_topk=32,index_n_heads=8,index_head_dim=16)
active=FixedTargetMetadataUpdater(tp_rank=0,tp_size=8,query_start_loc=query,groups=(group(1,True),group(4),group(128)),rotary=(RotaryBinding(full,zero,torch.zeros(96,1,1,16),torch.zeros(96,1,1,16)),),operators=ops)
private=active.make_scratch()
assert active.rotary[0].target_cos.data_ptr()!=private.rotary[0].target_cos.data_ptr()
assert all(a.sas_metadata.data_ptr()!=b.sas_metadata.data_ptr() for a,b in zip(active.groups,private.groups))
state=SimpleNamespace(target_positions=torch.arange(96,dtype=torch.int64),target_seq_lens=torch.full((12,),96,dtype=torch.int32))
private.update(state)
active.commit_from(private)
expected=FixedTargetMetadataUpdater(tp_rank=0,tp_size=8,query_start_loc=query,groups=(group(1,True),group(4),group(128)),rotary=(RotaryBinding(full,zero,torch.zeros(96,1,1,16),torch.zeros(96,1,1,16)),),operators=ops)
expected.update(state)
assert torch.equal(active.rotary[0].target_cos,expected.rotary[0].target_cos)
assert torch.equal(active.groups[0].swa_slot_mapping,expected.groups[0].swa_slot_mapping)
for a,b in zip(active.groups,expected.groups):
 for field in ('seq_lens','input_positions','start_pos','local_query_start_loc','local_seq_lens','sas_metadata','qli_metadata'):
  x=getattr(a,field);y=getattr(b,field)
  assert (x is None and y is None) or torch.equal(x,y),field
print('private metadata generation and stable destination commit: pass')
