#!/usr/bin/env python3
"""Read recent pool key metadata using the SDK's existing metadata-only client."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    request = json.loads((base/'tests/long-per-tp-kv-abba-v1/wave-11-prime.json').read_text())['request']
    cid = json.loads((base/'pd-node32-v10/launch.json').read_text())['container_id']
    out = base/'tests/per-tp-key-metadata-v3'
    out.mkdir(exist_ok=False)
    code = '''import json,torch_npu
from memcache_hybrid import DistributedObjectStore
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import init_none_hash,hash_block_tokens
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata,PoolKey
payload=PAYLOAD
def plain(value,depth=0):
 if value is None or isinstance(value,(str,int,float,bool)):return value
 if depth>3:return str(type(value))
 if isinstance(value,dict):return {str(k):plain(v,depth+1) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [plain(v,depth+1) for v in value]
 return {n:plain(getattr(value,n),depth+1) for n in dir(value) if not n.startswith('_') and not callable(getattr(value,n))}
report={'npu_initialized_before':torch_npu.npu.is_initialized(),'variants':[], 'scope':'Metadata only: init_bm=False, exists and batch_get_key_info; no device binding, payload read or model execution.'}
store=DistributedObjectStore()
try:
 ret=store.init(0,init_bm=False);assert ret==0,ret
 for algorithm in ['sha256','sha256_cbor']:
  fn=get_hash_fn_by_name(algorithm);init_none_hash(fn);parent=None;hashes=[]
  for start in range(0,len(payload['prompt']),128):
   parent=hash_block_tokens(fn,parent,payload['prompt'][start:start+128],(payload['cache_salt'],) if start==0 else None);hashes.append(parent.hex())
  for family in ['default','c1','mixed']:
   keys=[PoolKey(KeyMetadata('GLM-5.3-w8a8c8',rank,0,0,cache_family=family),hashes[block]).to_string() for block in [0,15,30,31] for rank in range(16)]
   present=store.batch_is_exist(keys)
   row={'algorithm':algorithm,'family':family,'queried':len(keys),'present':sum(x==1 for x in present)}
   if row['present']:
    selected=[k for k,s in zip(keys,present) if s==1]
    row['keys']=selected;row['key_info']=[{'size':i.size(),'gva':i.gva_list(),'location':i.loc_list(),'media_type':i.type_list()} for i in store.batch_get_key_info(selected)];report['variants'].append(row)
   else:report['variants'].append(row)
 report['completed']=True
except Exception as exc:
 report['completed']=False;report['failure']=type(exc).__name__+': '+str(exc)
finally:
 report['close_return']=store.close()
 report['npu_initialized_after']=torch_npu.npu.is_initialized()
print('GLM53_RESULT_JSON_BEGIN\\n'+json.dumps(report,indent=2)+'\\nGLM53_RESULT_JSON_END')
'''.replace('PAYLOAD', repr(request))
    proc = subprocess.run(['docker', 'exec', '-i', cid, 'python3', '-'], input=code,
                          capture_output=True, text=True, timeout=120)
    (out/'stdout.txt').write_text(proc.stdout)
    (out/'stderr.txt').write_text(proc.stderr)
    report = {'returncode': proc.returncode, 'source_sha256': hashlib.sha256(code.encode()).hexdigest(),
              'stdout_sha256': hashlib.sha256(proc.stdout.encode()).hexdigest(),
              'stderr_sha256': hashlib.sha256(proc.stderr.encode()).hexdigest()}
    if 'GLM53_RESULT_JSON_BEGIN' in proc.stdout:
        report['metadata'] = json.loads(proc.stdout.split('GLM53_RESULT_JSON_BEGIN',1)[1].split('GLM53_RESULT_JSON_END',1)[0])
    (out/'script.py').write_text(code)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert proc.returncode == 0, proc.stderr[-2000:]


if __name__ == '__main__':
    main()
