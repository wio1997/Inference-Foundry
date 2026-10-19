#!/usr/bin/env python3
"""CPU-only exact profiler/pool allocation source; no framework import/NPU devices."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1445-sglang-long-checkpoint-v1'
IMAGES={43:'sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'}
CODE="import pathlib,json,hashlib\nroot=pathlib.Path('/work/benchmark/ais_bench/benchmark');files=[];listing=[];paths=[]\nfor p in root.rglob('*.py'):\n text=p.read_text();hits=[{'line':i+1,'text':l[:300]} for i,l in enumerate(text.splitlines()) if ('generate_tokens_speed' in l or ('time_points' in l and any(x in l for x in ['[1]','[0]','[-1]','diff(','TPOT','ttft','tpot'])))]\n if hits:listing.append({'path':str(p.relative_to(root)),'hits':hits});paths.append(p)\nbudget=2000000\nfor p in paths:\n raw=p.read_bytes()\n if len(raw)>budget:continue\n budget-=len(raw);files.append({'path':str(p.relative_to(root)),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()})\nprint('GLM53_AIS_METRIC_SOURCE='+json.dumps({'files':files,'listing':listing,'model_requests':0}))\n"
def capture(node):
 pod,host=b.NODES[node];image=IMAGES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']==f'172.16.12.{node}'
 code=re.sub(r'\bNODE\b',repr(node),CODE)
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=2g','--pids-limit=128','--label','project_id=glm53','--name',f'glm53-ais-metric-source-v1-20261017-{node}','-v','/data/wio/aisbench-offline-43-20261014:/work:ro','--entrypoint','/work/venv/bin/python',image,'-c',code]
 hostcode=r'''import socket,subprocess,sys
assert socket.gethostname()==HOST
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert r.returncode==0 and r.stdout.strip()==IMAGE
import pathlib,json,hashlib,time
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1435-sglang-long-semantic-v1');out={'snapshot_monotonic_ns':time.monotonic_ns(),'files':{}}
for name in ['semantic-progress.json','semantic-result.json','semantic-phase-status.json','metadata-gate-0.json','metadata-gate-1.json']:
 p=root/name
 if p.is_file():
  raw=p.read_bytes();out['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data':json.loads(raw)}
for name in ['client-stdout.txt','client-stderr.txt']:
 p=root/name
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));raw=f.read()
  out['files'][name]={'full_bytes':p.stat().st_size,'tail':raw.decode(errors='replace')}
name='glm53-sglang-long-semantic-v1-20261017-43';r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
if r.returncode==0:
 d=json.loads(r.stdout)[0];assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1435-sglang-long-semantic-v1' and d['Image']==IMAGE
 out['CPUclient']={'id':d['Id'],'state':d['State']}
else:out['CPUclient_inspect_rc']=r.returncode
print('GLM53_LONG_CHECKPOINT='+json.dumps(out),flush=True)
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=160);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
'''
 for k,v in [('HOST',host),('IMAGE',image),('ARGS',args)]:hostcode=re.sub(r'\b'+k+r'\b',lambda _:repr(v),hostcode)
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],hostcode,timeout=210)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'model_requests':0,'npu_allocations':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(capture,n):n for n in IMAGES}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/f'source-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
