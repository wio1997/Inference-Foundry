# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU expert-row cache; explicit local ownership, no collectives or NPU work."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import torch

@dataclass(frozen=True)
class ExpertSliceKey:
 layer:int
 expert:int
 projection:str
 start:int
 stop:int

@dataclass(frozen=True)
class ExpertSlice:
 key:ExpertSliceKey
 recipe:str
 weight:torch.Tensor
 scale:torch.Tensor
 offset:torch.Tensor
 payload_bytes:int

class RankExpertHostCache:
 """Synchronous prefetch with LRU eviction; returns defensive CPU copies.

 max_bytes bounds resident tensor payload only, not Python/allocator overhead,
 transient reader buffers or caller-owned returned copies. No async prefetch,
 NPU execution, shared-expert handling or cross-rank dispatch is implied.
 """
 def __init__(self,reader,*,rank,world_size,max_bytes):
  if type(world_size) is not int or world_size not in (1,2,4,8,16) or type(rank) is not int or not 0<=rank<world_size:raise ValueError('supported explicit rank/world required')
  if type(max_bytes) is not int or max_bytes<1:raise ValueError('positive host payload budget required')
  text=reader.config['text_config'];self._count=text['n_routed_experts'];self._layers=text['mlp_layer_types'];self.reader=reader;self.rank=rank;self.world_size=world_size;self.max_bytes=max_bytes
  self._entries=OrderedDict();self._used=0;self._closed=False;self._lock=RLock();self._hits=0;self._misses=0
 def _plan(self,key):
  if not isinstance(key,ExpertSliceKey) or any(type(x) is not int for x in (key.layer,key.expert,key.start,key.stop)) or not 0<=key.layer<len(self._layers) or self._layers[key.layer]!='sparse' or not 0<=key.expert<self._count or key.expert%self.world_size!=self.rank or key.projection not in ('gate_proj','up_proj','down_proj'):raise ValueError('invalid or non-owned global expert slice')
  prefix=f'model.language_model.layers.{key.layer}.mlp.experts.{key.expert}.{key.projection}'
  names=[prefix+'.weight',prefix+'.weight_scale',prefix+'.weight_offset'];desc=[self.reader.descriptor(n)[2] for n in names]
  if any(self.reader.quant.get(n)!='W8A8_DYNAMIC' for n in names):raise ValueError('explicit W8A8_DYNAMIC metadata required')
  w,s,o=desc
  if w['dtype']!='I8' or len(w['shape'])!=2 or s['dtype']!='F32' or o['dtype']!='F32' or s['shape']!=[w['shape'][0],1] or o['shape']!=s['shape']:raise ValueError('invalid weight/scale/offset descriptor')
  if not 0<=key.start<key.stop<=w['shape'][0]:raise ValueError('invalid bounded row range')
  rows=key.stop-key.start;size=rows*(w['shape'][1]+8)
  if size>self.max_bytes or max(rows*w['shape'][1],rows*4)>self.reader.max_read_bytes:raise ValueError('slice exceeds host or reader byte budget')
  return names,w['shape'][1],size
 def _ensure_open(self):
  if self._closed:raise RuntimeError('cache is closed')
 def _load(self,key):
  names,width,size=self._plan(key)
  if key in self._entries:
   self._hits+=1;self._entries.move_to_end(key);return self._entries[key]
  self._misses+=1
  while self._used+size>self.max_bytes:
   _,victim=self._entries.popitem(last=False);self._used-=victim.payload_bytes
  values=[]
  for name,dtype in zip(names,('I8','F32','F32'),strict=True):
   tensor,actual=self.reader.row_slice(name,key.start,key.stop)
   if actual!=dtype or tensor.device.type!='cpu':raise ValueError('reader returned unexpected dtype/device')
   values.append(tensor)
  w,s,o=values;rows=key.stop-key.start
  if w.dtype!=torch.int8 or w.shape!=(rows,width) or s.dtype!=torch.float32 or o.dtype!=torch.float32 or s.shape!=(rows,1) or o.shape!=s.shape or not bool(torch.isfinite(s).all() and (s>0).all() and torch.isfinite(o).all() and (o==0).all()):raise ValueError('invalid W8A8 raw payload')
  actual=sum(t.numel()*t.element_size() for t in values)
  if actual!=size:raise ValueError('payload accounting mismatch')
  value=ExpertSlice(key,'W8A8_DYNAMIC',w,s,o,actual);self._entries[key]=value;self._used+=actual;return value
 def prefetch(self,keys):
  with self._lock:
   self._ensure_open()
   for key in keys:self._load(key)
 def read(self,key):
  with self._lock:
   self._ensure_open();v=self._load(key)
   return ExpertSlice(v.key,v.recipe,v.weight.clone(),v.scale.clone(),v.offset.clone(),v.payload_bytes)
 def stats(self):
  with self._lock:return {'rank':self.rank,'world_size':self.world_size,'used_bytes':self._used,'max_bytes':self.max_bytes,'keys_lru':tuple(self._entries),'hits':self._hits,'misses':self._misses,'closed':self._closed}
 def close(self):
  with self._lock:self._entries.clear();self._used=0;self._closed=True
