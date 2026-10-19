#!/usr/bin/env python3
"""Read exact-image hashing/graph/MTP source and owned Meta audit, no model calls."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0690-vllm-long-contract-v1'
CID='7c5a8fa28d185ff0aef35a1330e737934adc0d6cdb232919e2e05901c9f866d4'
META='af9a8410d2e1ce0d98045001b5e3628d048d96666a25bc0a1100a545d88c0157'
SOURCE_CODE=r'''
import pathlib,json,hashlib
v=pathlib.Path('/vllm-workspace/vllm');a=pathlib.Path('/vllm-workspace/vllm-ascend')
paths=[v/'vllm/utils/hashing.py',v/'vllm/v1/core/kv_cache_utils.py',v/'vllm/v1/request.py',a/'vllm_ascend/platform.py',a/'vllm_ascend/spec_decode/eagle_proposer.py',a/'vllm_ascend/spec_decode/mtp_proposer.py',a/'vllm_ascend/distributed/kv_transfer/utils/utils.py']
out={'files':[],'spec_files':[str(p) for p in (a/'vllm_ascend/spec_decode').rglob('*.py')]};budget=850000
for p in paths:
 if not p.is_file():continue
 data=p.read_bytes()
 if len(data)>budget:continue
 budget-=len(data);out['files'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()})
print(json.dumps(out))
'''
META_CODE=r'''
import pathlib,json,hashlib
root=pathlib.Path('/var/log/memcache_hybrid/logs');out=[]
for name in ['mmc-meta.log','mmc-meta-audit.log']:
 p=root/name
 if not p.is_file():continue
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-65536));data=f.read()
 out.append({'path':str(p),'file_bytes':p.stat().st_size,'tail_sha256':hashlib.sha256(data).hexdigest(),'tail':data.decode(errors='replace')})
print(json.dumps(out))
'''


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[30]
 m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert m['rc']==0 and m['stdout']=='172.16.12.30'
 code='''import socket,subprocess,json,sys
assert socket.gethostname()==EXPECTED
out={}
for cid,name,payload,label in [(CID,'glm53-vllm-correctness-v2-20261017-30',SOURCE_CODE,'model_source'),(META,'glm53-vllm-correctness-v2-meta-20261017-30',META_CODE,'meta_audit')]:
 result=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15);assert result.returncode==0
 data=json.loads(result.stdout)[0]
 assert data['Id']==cid and data['Name']=='/'+name and data['State']['Running']
 assert data['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' and data['Config']['Labels']['project_id']=='glm53'
 result=subprocess.run(['docker','exec',cid,'python3','-c',payload],capture_output=True,text=True,timeout=45)
 out[label]={'rc':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
print(json.dumps(out));sys.exit(0 if all(v['rc']==0 for v in out.values()) else 1)
'''
 import re
 for key,value in [('EXPECTED',hostname),('CID',CID),('META',META),('SOURCE_CODE',SOURCE_CODE),('META_CODE',META_CODE)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 result=b.invoke([command],code,timeout=140);(RUN/'source-30.json').write_text(json.dumps(result,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':result['rc'],'model_requests':0},indent=2)+'\n');print('long contract',result['rc'],flush=True)


if __name__=='__main__':main()
