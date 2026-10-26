#!/usr/bin/env python3
"""Exact-image cache/rank/model source and installed MemCache API inspection."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0300-framework-source'
CODE=r'''
import os,sys,pathlib,hashlib,json,subprocess,importlib.util,traceback
out={'sources':[],'git':{},'sdk_files':[],'api':{}};budget=2600000
roots=['/vllm-workspace/vllm-ascend','/vllm-workspace/vllm','/sgl-workspace/sglang']
for root in roots:
 if not pathlib.Path(root).exists():continue
 r=subprocess.run(['git','-C',root,'rev-parse','HEAD'],capture_output=True,text=True,timeout=10);out['git'][root]={'rc':r.returncode,'head':r.stdout.strip(),'error':r.stderr[:500]}
 targets=[]
 if 'vllm-ascend' in root:
  targets+=list((pathlib.Path(root)/'vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store').rglob('*.py'))
  targets += [pathlib.Path(root)/p for p in ['vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py','vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py','vllm_ascend/ascend_config.py']]
 elif root.endswith('/vllm'):
  targets += [pathlib.Path(root)/p for p in ['vllm/model_executor/models/deepseek_v2.py','vllm/distributed/parallel_state.py','vllm/model_executor/layers/fused_moe/config.py']]
 else:
  targets += [pathlib.Path(root)/'python'/p for p in ['sglang/srt/layers/dp_attention.py','sglang/srt/distributed/parallel_state.py','sglang/srt/arg_groups/fields/speculative.py','sglang/srt/arg_groups/fields/moe.py','sglang/srt/arg_groups/fields/cache.py','sglang/srt/arg_groups/fields/disaggregation.py','sglang/srt/entrypoints/openai/protocol.py']]
 for p in targets:
  if not p.is_file() or p.stat().st_size>600000:continue
  data=p.read_bytes()
  if len(data)>budget:continue
  budget-=len(data);out['sources'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')})
for package in ['memfabric_hybrid','memcache_hybrid']:
 spec=importlib.util.find_spec(package)
 if spec is None:continue
 root=pathlib.Path(spec.origin).parent
 targets=list(root.glob('VERSION'))+list(root.rglob('smem_trans*.h'))+list(root.rglob('smem_bm_def.h'))
 for p in targets:
  data=p.read_bytes();out['sdk_files'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')[:80000]})
try:
 import memcache_hybrid as mc
 for name in ['DistributedObjectStore','LocalConfig','MetaConfig','MetaService','KeyInfo']:
  cls=getattr(mc,name);out['api'][name]={}
  for attr in dir(cls):
   if attr.startswith('_') and attr!='__init__':continue
   member=getattr(cls,attr);out['api'][name][attr]={'doc':getattr(member,'__doc__',None),'repr':repr(member)[:300]}
except Exception:out['api_error']=traceback.format_exc()
print('GLM53_SOURCE_JSON='+json.dumps(out),flush=True)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 for n,image in [(30,'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'),(34,'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')]:
  pod,hostname=b.NODES[n]
  mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
  assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(n)
  args=['docker','run','--rm','--pull=never','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=256m','--network=none','--pids-limit=128','--memory=3g','--cpus=2','--label','project_id=glm53','--name','glm53-framework-source-20261017-'+str(n),'-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','timeout',image,'120','python3','-c',CODE]
  hc='import subprocess,socket,sys\nassert socket.gethostname()=='+repr(hostname)+'\nr=subprocess.run('+repr(args)+',capture_output=True,text=True,timeout=140)\nprint(r.stdout,end="");print(r.stderr,file=sys.stderr,end="");sys.exit(r.returncode)\n'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  r=b.invoke([command],hc,timeout=180);(RUN/('source-%s.json'%n)).write_text(json.dumps(r,indent=2));print(n,r['rc'],len(r['stdout']),flush=True)
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'npu_compute_requests':0},indent=2))
if __name__=='__main__':main()
