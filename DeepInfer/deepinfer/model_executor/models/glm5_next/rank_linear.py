# SPDX-License-Identifier: Apache-2.0
"""Explicit bounded output-row W8A8 projection seam; not a TP model runner."""
from __future__ import annotations
import hashlib
import torch,torch.distributed as dist
from torch import nn

class RankW8A8OutputLinear(nn.Module):
 """Each rank loads contiguous global output rows; replicated inputs are required.

 Four-rank mode gathers BF16 local outputs in rank order. No expert routing,
 row-parallel reductions, bias, uneven partitions or model integration implied.
 """
 def __init__(self,reader,prefix,device,*,rank,world_size,row_start,row_stop,max_output_rows,group=None):
  super().__init__()
  if torch.device(device).type!='npu':raise ValueError('explicit NPU device required')
  if type(world_size) is not int or world_size not in (1,4) or type(rank) is not int or not 0<=rank<world_size:raise ValueError('supported explicit rank/world required')
  if world_size==4:
   if not dist.is_initialized() or dist.get_world_size(group)!=world_size or dist.get_rank(group)!=rank:raise ValueError('explicit initialized matching process group required')
  elif dist.is_initialized():raise ValueError('one-rank path requires no initialized group')
  self.rank,self.world_size,self.group,self.row_start,self.row_stop=rank,world_size,group,row_start,row_stop
  self.device=torch.device(device)
  # Reserve the control flag before loading a local shard. All live ranks
  # must agree that initialization succeeded before any proceeds to forward.
  status=torch.zeros((1,),dtype=torch.int32,device=self.device) if world_size==4 else None
  failure=None
  try:
   if any(type(v) is not int for v in (row_start,row_stop,max_output_rows)) or not 0<=row_start<row_stop or max_output_rows<1 or row_stop-row_start>max_output_rows:raise ValueError('invalid bounded output rows')
   if not isinstance(prefix,str) or not prefix:raise ValueError('nonempty projection prefix required')
   self._prefix_fingerprint=int.from_bytes(hashlib.sha256(prefix.encode('utf-8')).digest()[:8],'big',signed=True)
   name=prefix+'.weight';d=reader.descriptor(name)[2]
   if reader.quant.get(name)!='W8A8_DYNAMIC' or d['dtype']!='I8' or len(d['shape'])!=2 or row_stop>d['shape'][0]:raise ValueError('expected bounded W8A8_DYNAMIC matrix')
   count=row_stop-row_start
   if count%(world_size*16):raise ValueError('equal 16-row aligned shards required')
   width=count//world_size;first=row_start+rank*width;last=first+width
   self.local_start,self.local_stop=first,last;self.input_size=d['shape'][1]
   self.local=reader.load_linear(prefix,device,output_slice=slice(first,last))
   if self.local.output_size!=width or self.local.input_size!=self.input_size:raise ValueError('loader returned incorrect local geometry')
   if world_size==4:
    import torch_npu
    torch_npu.npu.synchronize()
  except Exception as exc:failure=exc
  if world_size==4:
   status.fill_(int(failure is not None));dist.all_reduce(status,op=dist.ReduceOp.MAX,group=group)
   if int(status.item()):raise RuntimeError('one or more ranks failed linear initialization') from failure
  elif failure is not None:raise failure
 def forward(self,x):
  valid=(isinstance(x,torch.Tensor) and x.device.type=='npu' and x.dtype==torch.bfloat16 and x.ndim==2 and 1<=x.shape[0]<=8 and x.shape[1]==self.input_size)
  if self.world_size==1:
   if not valid:raise ValueError('bounded BF16 NPU token matrix required')
   return self.local(x).contiguous()
  if not dist.is_initialized() or dist.get_world_size(self.group)!=self.world_size or dist.get_rank(self.group)!=self.rank:raise RuntimeError('rank group changed after construction')
  # Exchange fixed-size status and geometry before any variable-size collective.
  # This diagnostic correctness path requires identical replicated inputs.
  meta=torch.tensor([0,x.shape[0],x.shape[1],self.row_start,self.row_stop,self._prefix_fingerprint] if valid else [1,0,0,0,0,0],dtype=torch.int64,device=self.device)
  metadata=[torch.empty_like(meta) for _ in range(self.world_size)];dist.all_gather(metadata,meta,group=self.group)
  if any(int(item[0].item()) for item in metadata) or not all(torch.equal(metadata[0],item) for item in metadata[1:]):raise ValueError('rank linear input preflight failed or geometry differs')
  # Preallocate the control flag before the potentially failing input buffers.
  # Synchronous local preparation failures must be seen by every live rank
  # before any rank enters the variable-sized input gather.
  status=torch.zeros((1,),dtype=torch.int32,device=self.device)
  prepare_failure=None;value=None;inputs=None
  try:
   value=x.contiguous();inputs=[torch.empty_like(value) for _ in range(self.world_size)]
   import torch_npu
   torch_npu.npu.synchronize()
  except Exception as exc:prepare_failure=exc
  status.fill_(int(prepare_failure is not None))
  dist.all_reduce(status,op=dist.ReduceOp.MAX,group=self.group)
  if int(status.item()):raise RuntimeError('one or more ranks failed linear input preparation') from prepare_failure
  dist.all_gather(inputs,value,group=self.group)
  if not all(torch.equal(inputs[0],item) for item in inputs[1:]):raise ValueError('rank linear inputs differ')
  failure=None;local=None;parts=None
  try:
   local=self.local(value).contiguous()
   if local.shape!=(value.shape[0],self.local_stop-self.local_start) or local.dtype!=value.dtype or local.device!=value.device:raise ValueError('rank linear output geometry differs')
   parts=[torch.empty_like(local) for _ in range(self.world_size)]
   # Surface synchronous device errors before the coordinated status exchange.
   import torch_npu
   torch_npu.npu.synchronize()
  except Exception as exc:failure=exc
  status=torch.tensor([int(failure is not None)],dtype=torch.int32,device=self.device)
  dist.all_reduce(status,op=dist.ReduceOp.MAX,group=self.group)
  if int(status.item()):raise RuntimeError('one or more ranks failed local linear projection') from failure
  dist.all_gather(parts,local,group=self.group)
  return torch.cat(parts,dim=-1)
