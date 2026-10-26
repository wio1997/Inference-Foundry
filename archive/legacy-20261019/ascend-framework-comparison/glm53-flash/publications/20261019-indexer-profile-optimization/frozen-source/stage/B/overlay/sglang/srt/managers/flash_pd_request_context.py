# SPDX-License-Identifier: Apache-2.0
"""Immutable actual Scheduler request-slot and transfer-room diagnostic tags."""
import contextvars,functools,json,os,pathlib
_FORWARD=contextvars.ContextVar('flash_actual_forward_requests',default=None)
_TRANSFER=contextvars.ContextVar('flash_actual_transfer_room',default=None)
_ROOT=pathlib.Path('/run-evidence')
_BOOK=None

def enabled():return (_ROOT/'request-context-enabled').is_file()
def _book():
 global _BOOK
 if _BOOK is None:
  v=json.loads((_ROOT/'request-contexts.json').read_text())
  if v.get('project_id')!='glm53-flash':raise ValueError('Invalid Flash request context book')
  byid=v['by_id'];byroom=v['by_room']
  for rid,t in byid.items():
   if rid!=t['rid'] or t['project_id']!='glm53-flash' or type(t['input_length']) is not int or not 0<t['input_length']<=133120 or byroom[str(t['bootstrap_room'])]!=t:
    raise ValueError('Invalid immutable Flash RID/room mapping')
  _BOOK=v
 return _BOOK

def with_batch_tags(original):
 @functools.wraps(original)
 def wrapped(self,batch,*args,**kwargs):
  if not enabled():return original(self,batch,*args,**kwargs)
  tags={};reqs=list(batch.reqs)
  if not reqs:raise ValueError('Unexpected empty diagnostic model batch')
  for req in reqs:
   tag=dict(_book()['by_id'][req.rid]);slot=int(req.kv.req_pool_idx)
   if slot<0 or slot in tags or len(req.origin_input_ids)!=tag['input_length'] or int(req.bootstrap_room)!=tag['bootstrap_room']:
    raise ValueError('Actual Scheduler RID/slot/room metadata does not match frozen request')
   tags[slot]=tag
  token=_FORWARD.set(tags)
  try:
   from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
   import torch
   append('actual_forward_batch',device=torch.npu.current_device(),role=os.environ['FLASH_PD_ROLE'],mode=str(batch.forward_mode),req_slots=list(tags),requests=list(tags.values()))
   return original(self,batch,*args,**kwargs)
  finally:_FORWARD.reset(token)
 return wrapped

def forward_tags():
 if not enabled():return None
 v=_FORWARD.get()
 if v is None:raise ValueError('Missing actual Scheduler forward request context')
 return {slot:dict(tag) for slot,tag in v.items()}

def set_transfer_room(room):
 if not enabled():return
 room=int(room)
 if str(room) not in _book()['by_room']:raise ValueError('Unknown actual transfer room')
 _TRANSFER.set(room)

def transfer_tag():
 if not enabled():return None
 room=_TRANSFER.get()
 if room is None:raise ValueError('Missing actual TransferKVChunk.room context')
 return dict(_book()['by_room'][str(room)])
