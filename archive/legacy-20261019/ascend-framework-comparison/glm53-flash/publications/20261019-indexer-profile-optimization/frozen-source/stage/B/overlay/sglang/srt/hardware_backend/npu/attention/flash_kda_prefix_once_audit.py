"""Bounded P-only warm checkpoint/restore hashes for the single functional case.

No per-token tracer. Exactly one warm checkpoint and one initial restored-state
comparison per KDA layer/rank. Disable entirely before formal performance.
"""
import os,hashlib,functools
_WARM={}
_RESTORED=set()

def _observe(fn):
 @functools.wraps(fn)
 def wrapped(backend,layer,batch):
  if os.environ.get('FLASH_KDA_PREFIX_ONCE_AUDIT')!='1' or os.environ.get('FLASH_PD_ROLE')!='Prefill':return None
  try:return fn(backend,layer,batch)
  except Exception as error:
   # Missing/failed proof blocks the independent acceptance. Observational
   # metadata errors must not kill an otherwise healthy native model worker.
   try:
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    import torch
    append('single_case_prefix_observer_error',device=torch.npu.current_device(),role='Prefill',layer_id=int(layer.layer_id),phase=fn.__name__,error=repr(error),functional_only=True)
   except Exception:pass
   return None
 return wrapped

def _tags():
 if os.environ.get('FLASH_KDA_PREFIX_ONCE_AUDIT')!='1' or os.environ.get('FLASH_PD_ROLE')!='Prefill':return None
 from sglang.srt.managers.flash_pd_request_context import forward_tags
 tags=forward_tags()
 if tags is None:return None
 if len(tags)!=1:raise ValueError('Bounded prefix observer requires the sole functional request')
 return next(iter(tags.values()))

def _ints(x):
 if x is None:return []
 return x.detach().cpu().reshape(-1).tolist() if hasattr(x,'detach') else list(x)

def _state(backend,layer_id,slot):
 import torch
 cache=backend.req_to_token_pool.mamba2_layer_cache(layer_id)
 result={}
 for name,tensor in [('conv',cache.conv[0]),('SSM',cache.temporal)]:
  if not 0<int(slot)<tensor.shape[0]:raise ValueError('Invalid actual checkpoint state slot')
  cpu=tensor[int(slot)].detach().contiguous().cpu()
  if not bool(cpu.isfinite().all()):raise ValueError('Nonfinite actual prefix state')
  result[name]={'dtype':str(cpu.dtype),'shape':list(cpu.shape),'bytes':cpu.numel()*cpu.element_size(),'sha256':hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}
 return result

@_observe
def after(backend,layer,batch):
 tag=_tags()
 if tag is None or tag['purpose']!='warm_prefix' or _ints(batch.seq_lens_cpu)!=[74400]:return
 layer_id=int(layer.layer_id)
 if layer_id in _WARM:raise ValueError('Duplicate final warm checkpoint')
 fm=backend.forward_metadata;h_dst=_ints(fm.track_ssm_h_dst);final_dst=_ints(fm.track_ssm_final_dst)
 if len(h_dst)+len(final_dst)!=1:raise ValueError('Missing unique actual warm checkpoint destination')
 if h_dst:
  chunk=_ints(fm.track_chunk_idx);prefix=_ints(batch.extend_prefix_lens_cpu)
  if len(chunk)!=1 or chunk[0]<0 or len(prefix)!=1:raise ValueError('Missing actual pre-chunk boundary metadata')
  boundary=prefix[0]+chunk[0]*64;slot=h_dst[0];kind='FP32 pre-chunk snapshot'
 else:boundary=74400;slot=final_dst[0];kind='native final-state snapshot'
 state=_state(backend,layer_id,slot);_WARM[layer_id]={'boundary':boundary,'state':state}
 from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
 import torch
 append('single_case_warm_checkpoint',device=torch.npu.current_device(),role='Prefill',layer_id=layer_id,request_id=tag['rid'],absolute_boundary=boundary,state_slot=int(slot),snapshot_kind=kind,state=state,functional_only=True)

@_observe
def before(backend,layer,batch):
 tag=_tags()
 if tag is None or tag['purpose']!='same_fixed_case_functional':return
 layer_id=int(layer.layer_id)
 if layer_id in _RESTORED:return
 _RESTORED.add(layer_id)
 prefix=_ints(batch.extend_prefix_lens_cpu)
 if len(prefix)!=1 or prefix[0]<128:raise ValueError('Actual warm prefix reuse absent')
 expected=_WARM.get(layer_id)
 if expected is None or expected['boundary']!=prefix[0]:raise ValueError('Restored prefix label differs from the actual captured checkpoint boundary')
 slots=_ints(backend.forward_metadata.mamba_cache_indices)
 if len(slots)!=1:raise ValueError('Wrong actual restored-state cardinality')
 actual=_state(backend,layer_id,slots[0]);passed=actual==expected['state']
 from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
 import torch
 append('single_case_warm_restore',device=torch.npu.current_device(),role='Prefill',layer_id=layer_id,request_id=tag['rid'],absolute_boundary=prefix[0],state_slot=int(slots[0]),state=actual,expected_state=expected['state'],passed=passed,functional_only=True)
 if not passed:raise ValueError('Native restored conv/SSM bytes differ from warm checkpoint')
