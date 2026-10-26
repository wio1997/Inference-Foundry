#!/usr/bin/env python3
"""Exact-image cache/rank/model source and installed MemCache API inspection."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0740-sglang-config-source-v1'
CODE=r'''
import pathlib,json,hashlib
root=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt');out={'files':[],'paths':[]};budget=4500000
candidates=[root/'server_args.py']+sorted((root/'arg_groups').rglob('*.py'))+sorted((root/'runtime_context').rglob('*.py'))+sorted((root/'disaggregation').rglob('*.py'))
for p in candidates:
 if not p.is_file():continue
 data=p.read_bytes();out['paths'].append({'path':str(p),'bytes':len(data)})
 include=('disaggregation' in str(p) or 'runtime_context' in str(p) or p.name=='server_args.py' or any(k in data for k in [b'speculative_num_draft_tokens',b'hicache',b'dp_size',b'ep_size',b'context_length',b'mem_fraction_static',b'chunked_prefill_size',b'quantization']))
 if not include or len(data)>budget or len(data)>650000:continue
 budget-=len(data);out['files'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()})
print('GLM53_SOURCE_JSON='+json.dumps(out),flush=True)

'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 for n,image in [(34,'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')]:
  pod,hostname=b.NODES[n]
  mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
  assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(n)
  args=['docker','run','--rm','--pull=never','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=256m','--network=none','--pids-limit=128','--memory=3g','--cpus=2','--label','project_id=glm53','--name','glm53-sglang-config-source-v1-20261017-'+str(n),'-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','--entrypoint','timeout',image,'120','python3','-c',CODE]
  hc='import subprocess,socket,sys\nassert socket.gethostname()=='+repr(hostname)+'\nr=subprocess.run('+repr(args)+',capture_output=True,text=True,timeout=140)\nprint(r.stdout,end="");print(r.stderr,file=sys.stderr,end="");sys.exit(r.returncode)\n'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  r=b.invoke([command],hc,timeout=180);(RUN/('source-%s.json'%n)).write_text(json.dumps(r,indent=2));print(n,r['rc'],len(r['stdout']),flush=True)
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'npu_compute_requests':0},indent=2))
if __name__=='__main__':main()
