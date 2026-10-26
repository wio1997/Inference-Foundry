#!/usr/bin/env python3
"""Read exact-image CLI/proxy/shutdown contracts before model deployment."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0340-integration-contract'
CODE=r'''
import pathlib,json,hashlib,subprocess,importlib.metadata as md
root=pathlib.Path('/vllm-workspace/vllm');asc=pathlib.Path('/vllm-workspace/vllm-ascend')
out={'files':[],'snippets':[],'client_packages':[]};budget=2500000
targets=[root/p for p in ['vllm/engine/arg_utils.py','vllm/distributed/kv_transfer/kv_connector/v1/base.py','vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py','vllm/config/compilation.py']]
targets+=[p for p in (root/'vllm/entrypoints/openai').rglob('protocol.py') if 'class CompletionRequest' in p.read_text()]
targets+=[p for p in (root/'examples/disaggregated_prefill_v1').rglob('*.py') if 'proxy' in p.name]
targets+=[asc/'vllm_ascend/worker/worker.py']
for p in targets:
 if not p.is_file():continue
 data=p.read_bytes()
 if len(data)>budget:continue
 budget-=len(data);out['files'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')})
for search in [root/'vllm/v1/engine',root/'vllm/v1/worker',root/'vllm/v1/core',root/'vllm/distributed/kv_transfer']:
 for p in search.rglob('*.py'):
  text=p.read_text();lines=text.splitlines();indices=set()
  for i,line in enumerate(lines):
   if 'last_token_id' in line or ('shutdown' in line and ('kv_' in text or 'connector' in line)):
    indices.update(range(max(0,i-8),min(len(lines),i+18)))
  if indices:
   out['snippets'].append({'path':str(p),'sha256':hashlib.sha256(text.encode()).hexdigest(),'lines':[[i+1,lines[i]] for i in sorted(indices)][:250]})
for d in md.distributions():
 name=d.metadata.get('Name','')
 if any(t in name.lower() for t in ['ais','bench','aiohttp','httpx','fastapi','uvicorn']):out['client_packages'].append({'name':name,'version':d.version})
print(json.dumps(out))
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[30]
 m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.30'
 args=['docker','run','--rm','--pull=never','--read-only','--network=none','--pids-limit=32','--memory=1g','--cpus=1','--label','project_id=glm53','--name','glm53-integration-contract-20261017-30','--entrypoint','timeout','sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83','60','python3','-c',CODE]
 hc='import subprocess,socket,sys\nassert socket.gethostname()=='+repr(hostname)+'\nr=subprocess.run('+repr(args)+',capture_output=True,text=True,timeout=80)\nprint(r.stdout,end="");print(r.stderr,file=sys.stderr,end="");sys.exit(r.returncode)\n'
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 r=b.invoke([command],hc,timeout=110);(RUN/'source-30.json').write_text(json.dumps(r,indent=2));print(r['rc'],len(r['stdout']),flush=True)
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'npu_compute_requests':0},indent=2))
if __name__=='__main__':main()
