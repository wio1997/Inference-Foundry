"""Whole original streaming/base/session classes plus exact facade match body.

The row pool/allocator below are CPU ownership ledgers, not accelerator pools.
"""
from L035_caller_components_v3 import *
from L034_extension_components_v4 import shell

load('python/sglang/srt/utils/common.py',functions=['ceil_align'])
load('python/sglang/srt/mem_cache/base_prefix_cache.py',classes=['CacheRequestOutcome','PrefixCacheTrait','BasePrefixCache','InitLoadBackParams'])
load('python/sglang/srt/mem_cache/common.py',functions=['coalesce_ranges','free_kv_row_segments'])
common=shell('sglang.srt.mem_cache.common')
common.coalesce_ranges=ns['coalesce_ranges'];common.free_kv_row_segments=ns['free_kv_row_segments']
schedule_module=shell('sglang.srt.managers.schedule_batch');schedule_module.FINISH_ABORT=ns['FINISH_ABORT']
branch=NS(npu=False)
ns['is_npu']=lambda:branch.npu
load('python/sglang/srt/session/streaming_session.py',classes=['_VirtualNode','SessionSlot','StreamingSession'],functions=['_is_streaming'])
# Keep the entire original decorator expression, but explicitly replace rank
# consensus with an identity decorator: this fixture has no process groups.
ns['rank_consensus']=lambda **kw:lambda f:f
load('python/sglang/srt/mem_cache/unified_radix_cache.py',methods={'UnifiedRadixCache':['match_prefix']})

class CPUAllocatorLedger:
 def __init__(self,page):self.page_size=page;self.freed=set();self.calls=[];self.owned=set()
 def own(self,indices):
  pages=set((indices//self.page_size).tolist());assert not self.owned;self.owned=pages
 def _free(self,segments,kind):
  pages=set();parts=[]
  for values,start in segments:
   part=set((values//self.page_size).tolist());assert not (pages & part),'duplicate page in one free';pages.update(part);parts.append({'start':start,'count':values.numel()})
  assert not (self.freed & pages),'double free';assert pages<=self.owned,'unowned page free';self.freed.update(pages);self.calls.append({'kind':kind,'parts':parts,'pages':sorted(pages)})
 def free_segments(self,segments):self._free(segments,'full-and-swa')
 def free_full_segments(self,segments):self._free(segments,'full-only')
 def max_new_tokens_for_memory(self,input_len,max_new_tokens,**kw):return max_new_tokens

class CPURowPool:
 def __init__(self,width):
  self.req_to_token=torch.arange(1024,1024+width,dtype=torch.int64).reshape(1,width);self.freed_rows=[];self.mamba_allocator=None
 def free(self,holder):
  idx=holder.kv.req_pool_idx;assert idx is not None and idx not in self.freed_rows;self.freed_rows.append(idx);holder.kv.req_pool_idx=None

class CoreCounter:
 def __init__(self,core):self.core=core;self.calls=0
 def match_prefix(self,params):self.calls+=1;return self.core.match_prefix(params)
 def __getattr__(self,name):return getattr(self.core,name)

class ComposedView(CoreView):
 match_prefix=ns['UnifiedRadixCache'].match_prefix
 def __init__(self,page,eagle,width):
  super().__init__(page,eagle);self.tree_core=CoreCounter(self.tree_core);self.disable=False;self._components_tuple=(self.comp,);self.linker=None;self.req_to_token_pool=CPURowPool(width);self.token_to_kv_pool_allocator=CPUAllocatorLedger(page);self.session=ns['StreamingSession'](inner=self);self.lock_releases=[];self.radix_releases=[]
 def _apply_cache_actions(self,actions):assert not actions,'fixture permits empty-match actions only'
 def dec_lock_ref(self,node,receipt,**kw):self.lock_releases.append({'node':node,'skip_swa':kw.get('skip_swa',False)});return ns['DecLockRefResult']()
 def release_session(self,sid):return self.session.release_session(sid)
 def release_radix_session(self,sid):self.radix_releases.append(sid)
 def ensure_session_generation(self,sid):raise AssertionError('radix-tag sessions disabled in streaming fixture')

def composed_scheduler(view,mode):
 s=scheduler(view,mode);s.session_controller=ns['SessionController'](view);return s

BOUNDARIES_L036=BOUNDARIES_L035+[
 'Whole original BasePrefixCache and StreamingSession classes with original coalesce_ranges/free_kv_row_segments; row tensors and page/free ownership are CPU ledgers, not actual allocator/NPU KV',
 'Original facade match_prefix body and original Session/SessionController plus Scheduler session branch executed; rank_consensus explicitly identity-decorated, no TP/DP groups or full facade constructor',
 'is_npu false/true selects original Python branch only; no accelerator is available. Tree-lock release is a callback ledger, no actual native/tree lock ownership assertion',
 'Synthetic finish3/output and extra allocation are lifecycle fixtures under unchanged128/256 budgets, not real generated outputs or a shortened serving benchmark; no timing/performance claim',
 'Non-streaming Core fallback runs actual empty Core/FULL; storage/backend, multimodal/Mamba and persistent generations remain unexecuted']
