#!/usr/bin/env python3
"""Reversible first eager prefill metadata/address fingerprint probe."""
import argparse,hashlib,json
from pathlib import Path
SRC=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
BKP=Path('/tmp/loop053_run206_runner_backup.py')
PATCH=r'''

# EXTREME_LOOP053_RUN206: read-only first88 prefill fingerprint.
if os.getenv('EXTREME_RUN206_OUT_DIR'):
    _extreme_run206_orig_forward = NPUModelRunner._model_forward
    def _extreme_run206_fingerprint(obj, path, depth, seen, budget, hashes):
        import hashlib as _hashlib
        if budget[0] <= 0:
            return {'truncated': True}
        budget[0] -= 1
        if obj is None or isinstance(obj, (bool,int,float,str)):
            return obj if not isinstance(obj,str) or len(obj)<160 else obj[:160]
        if isinstance(obj, torch.Tensor):
            result={'kind':'tensor','shape':list(obj.shape),'dtype':str(obj.dtype),'device':str(obj.device),
                    'stride':list(obj.stride()),'data_ptr':int(obj.data_ptr()),'numel':int(obj.numel())}
            keywords=('slot','block','seq','query','position','input_ids','start_pos','length','indices','index')
            if hashes[0]<64 and obj.numel()<=2048 and any(k in path.lower() for k in keywords) and obj.dtype in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8,torch.bool):
                try:
                    result['value_sha256']=_hashlib.sha256(obj.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    hashes[0]+=1
                except Exception as exc:
                    result['hash_error']=type(exc).__name__
            return result
        oid=id(obj)
        if oid in seen:return {'ref': True,'type':type(obj).__name__}
        if depth>=7:return {'type':type(obj).__name__,'depth_limit':True}
        seen.add(oid)
        if isinstance(obj,dict):
            keys=sorted(obj,key=lambda x:str(x))
            return {'kind':'dict','len':len(obj),'items':{str(k):_extreme_run206_fingerprint(obj[k],path+'.'+str(k),depth+1,seen,budget,hashes) for k in keys[:100]}}
        if isinstance(obj,(list,tuple)):
            return {'kind':type(obj).__name__,'len':len(obj),'items':[_extreme_run206_fingerprint(v,path+'['+str(i)+']',depth+1,seen,budget,hashes) for i,v in enumerate(obj[:24])]}
        attrs=getattr(obj,'__dict__',None)
        if isinstance(attrs,dict):
            keys=sorted(k for k in attrs if not k.startswith('__'))
            return {'kind':type(obj).__name__,'attrs':{k:_extreme_run206_fingerprint(attrs[k],path+'.'+k,depth+1,seen,budget,hashes) for k in keys[:80]}}
        return {'type':type(obj).__name__}
    def _extreme_run206_forward(self,num_tokens_padded,*args,**kwargs):
        import json as _json
        from pathlib import Path as _Path
        tagpath=os.getenv('EXTREME_RUN206_TAG_FILE')
        if not tagpath or not os.path.exists(tagpath):
            return _extreme_run206_orig_forward(self,num_tokens_padded,*args,**kwargs)
        ctx=get_forward_context()
        if str(getattr(ctx,'cudagraph_runtime_mode',''))!='NONE':
            return _extreme_run206_orig_forward(self,num_tokens_padded,*args,**kwargs)
        tag=_Path(tagpath).read_text().strip()
        rank=int(get_tp_group().rank_in_group)
        out=_Path(os.environ['EXTREME_RUN206_OUT_DIR'])/f'{tag}_rank{rank}.json'
        if out.exists():
            return _extreme_run206_orig_forward(self,num_tokens_padded,*args,**kwargs)
        budget=[5000];hashes=[0];seen=set()
        snapshot={'rank':rank,'tag':tag,'num_tokens_padded':int(num_tokens_padded),'num_actual_tokens_raw':repr(getattr(ctx,'num_actual_tokens',None)),'num_actual_tokens':int(getattr(ctx,'num_actual_tokens',None) or num_tokens_padded),'num_reqs':int(self.input_batch.num_reqs),'cudagraph_runtime_mode':str(getattr(ctx,'cudagraph_runtime_mode','')),
                  'args':_extreme_run206_fingerprint(args,'args',0,seen,budget,hashes),
                  'kwargs':_extreme_run206_fingerprint(kwargs,'kwargs',0,seen,budget,hashes),
                  'attn_metadata':_extreme_run206_fingerprint(getattr(ctx,'attn_metadata',None),'attn_metadata',0,seen,budget,hashes),
                  'nodes_remaining':budget[0],'small_integer_tensors_hashed':hashes[0]}
        result=_extreme_run206_orig_forward(self,num_tokens_padded,*args,**kwargs)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(_json.dumps(snapshot,separators=(',',':'))+'\n')
        return result
    NPUModelRunner._model_forward=_extreme_run206_forward
'''
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['install','restore']);p.add_argument('--record',required=True);a=p.parse_args()
 if a.action=='install':
  assert not BKP.exists();before=SRC.read_bytes();assert b'EXTREME_LOOP053_RUN206' not in before
  after=before+PATCH.encode();compile(after,str(SRC),'exec');BKP.write_bytes(before);SRC.write_bytes(after)
  r={'action':'install','original_sha256':sha(before),'patched_sha256':sha(after)}
 else:
  assert BKP.exists();before=SRC.read_bytes();assert b'EXTREME_LOOP053_RUN206' in before
  SRC.write_bytes(BKP.read_bytes());BKP.unlink();r={'action':'restore','patched_sha256':sha(before),'restored_sha256':sha(SRC.read_bytes())}
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
