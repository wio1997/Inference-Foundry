# SPDX-License-Identifier: Apache-2.0
"""Read-only KDA handoff trace keyed by actual Scheduler RID/slot metadata."""
import hashlib,os

def _ints(t):
 if t is None:return None
 return t.detach().cpu().reshape(-1).tolist() if hasattr(t,'detach') else list(t)

def snapshot(backend,layer,forward_batch,phase,active):
 if os.environ.get('FLASH_PD_STATE_AUDIT','1')=='0' or active is None:return None
 from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
 import torch
 role=os.environ['FLASH_PD_ROLE'];positions=_ints(forward_batch.positions)
 if len(positions)>128:raise ValueError('Flash qualification actual model token count exceeds128')
 indices=_ints(backend.forward_metadata.mamba_cache_indices);req_slots=_ints(forward_batch.req_pool_indices);seq_lens=_ints(forward_batch.seq_lens)
 if not req_slots or len(indices)!=len(req_slots) or len(seq_lens)!=len(req_slots) or len(set(req_slots))!=len(req_slots):raise ValueError('Invalid actual batched KDA metadata')
 starts=_ints(backend.forward_metadata.query_start_loc)
 if starts is None or len(starts)!=len(req_slots)+1 or starts[0]!=0 or starts[-1]!=len(positions):raise ValueError('Missing actual per-request KDA token boundaries')
 cache=backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
 def record(t,index):
  cpu=t[index].detach().contiguous().cpu();return {'shape':list(cpu.shape),'dtype':str(cpu.dtype),'sha256':hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),'nonfinite':int((~cpu.isfinite()).sum().item())}
 records=[]
 for j,(req_slot,slot,seq_len) in enumerate(zip(req_slots,indices,seq_lens)):
  tag=active[int(req_slot)];n=tag['input_length'];pos=positions[starts[j]:starts[j+1]]
  if role=='Prefill':
   if phase!='extend_after' or not pos or max(pos)!=n-1:continue
   expected_seq=n
  elif role=='Decode':
   if pos!=[n]:continue
   expected_seq=n+1
  else:raise ValueError('Unknown actual Flash PD role')
  if seq_len!=expected_seq or not 0<int(slot)<cache.temporal.shape[0]:raise ValueError('Unexpected actual batched KDA handoff position/slot')
  conv=record(cache.conv[0],slot);ssm=record(cache.temporal,slot);null_conv=record(cache.conv[0],0);null_ssm=record(cache.temporal,0)
  if any(v['nonfinite'] for v in [conv,ssm,null_conv,null_ssm]):raise ValueError('Nonfinite actual KDA state')
  records.append(append('kda_handoff',device=torch.npu.current_device(),role=role,phase=phase,tag=tag,layer=int(layer.layer_id),positions=pos,seq_lens=[seq_len],req_slots=[req_slot],state_slots=[slot],conv=conv,ssm=ssm,null_conv=null_conv,null_ssm=null_ssm,forward_mode=str(forward_batch.forward_mode),actual_batch_requests=len(req_slots),actual_batch_tokens=len(positions),actual_query_lengths=[b-a for a,b in zip(starts,starts[1:])]))
 return records
