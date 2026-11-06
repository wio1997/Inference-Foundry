# SPDX-License-Identifier: Apache-2.0
"""Versioned, bounded, owner-bound GLM mixed request snapshots (experimental)."""
from __future__ import annotations
import hashlib,hmac,io,json
import torch
from .cache import DSAState
from .kda import KDAState
from .request_state import MixedRequestState

_VERSION=2
_LEGACY_FIELDS={'version','config_sha256','request_id','tokens_seen','layers'}
_FIELDS=_LEGACY_FIELDS|{'layout_sha256'}
_LAYOUT_FIELDS=('latent_dim','index_dim','pool_size','topk_tokens','logical_block_size','lookahead')

def _fingerprint(config):
 return hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _layout_fingerprint(layout):
 values={name:getattr(layout,name) for name in _LAYOUT_FIELDS}
 if any(type(value) is not int for value in values.values()):raise ValueError('invalid snapshot layout')
 return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _bound(value,name):
 if type(value) is not int or value<1:raise ValueError(f'{name} must be a positive integer')
 return value

def _cpu(t):
 if not isinstance(t,torch.Tensor):raise ValueError('snapshot member must be a tensor')
 return t.detach().to(device='cpu').contiguous().clone()

@torch.inference_mode()
def serialize_snapshot(state,config,layout,*,max_tokens,max_archive_bytes):
 _bound(max_tokens,'max_tokens');_bound(max_archive_bytes,'max_archive_bytes')
 if not isinstance(state,MixedRequestState):raise ValueError('mixed request snapshot required')
 owned=MixedRequestState.capture(state.request_id,config,layout,state.layers)
 if owned.tokens_seen!=state.tokens_seen or owned.tokens_seen>max_tokens:raise ValueError('inconsistent or oversized snapshot')
 layers={}
 for spec in config.layers:
  s=owned.layers[spec.layer_id]
  if spec.attention=='linear_attention':layers[spec.layer_id]={'kind':'kda','tokens_seen':s.tokens_seen,'conv':[_cpu(t) for t in s.conv],'recurrent':_cpu(s.recurrent)}
  elif spec.attention=='deepseek_sparse_attention':layers[spec.layer_id]={'kind':'dsa','tokens_seen':s.tokens_seen,'latent':_cpu(s.latent),'compressed':_cpu(s.compressed),'tail':_cpu(s.tail),'tail_positions':_cpu(s.tail_positions)}
  else:raise ValueError('unsupported attention kind')
 payload={'version':_VERSION,'config_sha256':_fingerprint(config),'layout_sha256':_layout_fingerprint(layout),'request_id':owned.request_id,'tokens_seen':owned.tokens_seen,'layers':layers}
 out=io.BytesIO();torch.save(payload,out);data=out.getvalue()
 if len(data)>max_archive_bytes:raise ValueError('snapshot exceeds archive byte budget')
 return data

@torch.inference_mode()
def deserialize_snapshot(data,config,layout,*,expected_request_id,expected_tokens_seen,device,max_tokens,max_archive_bytes,allow_legacy_v1=False):
 if type(allow_legacy_v1) is not bool:raise ValueError('legacy snapshot opt-in must be boolean')
 _bound(max_tokens,'max_tokens');_bound(max_archive_bytes,'max_archive_bytes')
 if type(data) is not bytes or len(data)>max_archive_bytes:raise ValueError('snapshot bytes exceed bound or have invalid type')
 if not isinstance(expected_request_id,str) or not expected_request_id:raise ValueError('explicit expected owner required')
 if type(expected_tokens_seen) is not int or not 0<=expected_tokens_seen<=max_tokens:raise ValueError('invalid expected token count')
 payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
 if type(payload) is not dict or type(payload.get('version')) is not int:raise ValueError('unknown snapshot schema/version')
 if payload['version']==_VERSION:
  if set(payload)!=_FIELDS or payload['layout_sha256']!=_layout_fingerprint(layout):raise ValueError('snapshot layout or schema mismatch')
 elif payload['version']==1:
  if not allow_legacy_v1 or set(payload)!=_LEGACY_FIELDS:raise ValueError('legacy snapshot requires explicit unverified-layout opt-in')
 else:raise ValueError('unknown snapshot schema/version')
 if payload['config_sha256']!=_fingerprint(config) or payload['request_id']!=expected_request_id or type(payload['tokens_seen']) is not int or payload['tokens_seen']!=expected_tokens_seen:raise ValueError('snapshot config, owner or token count mismatch')
 raw=payload['layers'];ids={s.layer_id for s in config.layers}
 if type(raw) is not dict or set(raw)!=ids or any(type(i) is not int for i in raw):raise ValueError('snapshot layer IDs mismatch')
 layers={}
 for spec in config.layers:
  item=raw[spec.layer_id]
  if type(item) is not dict or type(item.get('tokens_seen')) is not int or item['tokens_seen']!=expected_tokens_seen:raise ValueError('layer metadata mismatch')
  if spec.attention=='linear_attention':
   if set(item)!={'kind','tokens_seen','conv','recurrent'} or item['kind']!='kda' or type(item['conv']) is not list or len(item['conv'])!=3:raise ValueError('invalid KDA schema')
   layers[spec.layer_id]=KDAState(tuple(item['conv']),item['recurrent'],item['tokens_seen'])
  elif spec.attention=='deepseek_sparse_attention':
   if set(item)!={'kind','tokens_seen','latent','compressed','tail','tail_positions'} or item['kind']!='dsa':raise ValueError('invalid DSA schema')
   layers[spec.layer_id]=DSAState(item['latent'],item['compressed'],item['tail'],item['tail_positions'],item['tokens_seen'])
  else:raise ValueError('unsupported attention kind')
 # Validate complete dtype, shapes, position tags, and ownership before copying to device.
 checked=MixedRequestState.capture(expected_request_id,config,layout,layers)
 if checked.tokens_seen!=expected_tokens_seen:raise ValueError('snapshot token count mismatch')
 moved={}
 for spec in config.layers:
  s=checked.layers[spec.layer_id]
  if spec.attention=='linear_attention':moved[spec.layer_id]=KDAState(tuple(t.to(device) for t in s.conv),s.recurrent.to(device),s.tokens_seen)
  else:moved[spec.layer_id]=DSAState(s.latent.to(device),s.compressed.to(device),s.tail.to(device),s.tail_positions.to(device),s.tokens_seen)
 return MixedRequestState.capture(expected_request_id,config,layout,moved)


def deserialize_snapshot_verified(data,config,layout,*,expected_sha256,expected_request_id,expected_tokens_seen,device,max_tokens,max_archive_bytes,allow_legacy_v1=False):
 """Restore only when bytes match a separately trusted producer digest.

 This detects mismatches; it does not authenticate the producer or digest.
 The digest must be persisted/transmitted through a trusted channel.
 """
 _bound(max_archive_bytes,'max_archive_bytes')
 if type(data) is not bytes or len(data)>max_archive_bytes:raise ValueError('snapshot bytes exceed bound or have invalid type')
 if type(expected_sha256) is not str or len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):raise ValueError('expected archive SHA256 must be lowercase hexadecimal')
 if not hmac.compare_digest(hashlib.sha256(data).hexdigest(),expected_sha256):raise ValueError('archive content digest mismatch')
 return deserialize_snapshot(data,config,layout,expected_request_id=expected_request_id,expected_tokens_seen=expected_tokens_seen,device=device,max_tokens=max_tokens,max_archive_bytes=max_archive_bytes,allow_legacy_v1=allow_legacy_v1)
