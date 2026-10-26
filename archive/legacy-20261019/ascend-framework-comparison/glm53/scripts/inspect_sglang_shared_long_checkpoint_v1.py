#!/usr/bin/env python3
"""One read-only active-long-job checkpoint, no inference/cancellation."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1743-sglang-shared-long-checkpoint-v1'
CODE=r'''import socket,pathlib,json,hashlib,subprocess,time
assert socket.gethostname()==HOSTNAME
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1740-sglang-shared-long-semantic-v4')
out={'project_id':'glm53','snapshot_monotonic_ns':time.monotonic_ns(),'files':{}}
for name in ['semantic-progress.json','semantic-result.json','semantic-phase-status.json','metadata-gate-0.json','metadata-gate-1.json']:
 p=root/name
 if p.is_file():
  raw=p.read_bytes();out['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode()}
for name in ['client-stdout.txt','client-stderr.txt']:
 p=root/name
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));raw=f.read()
  out['files'][name]={'full_bytes':p.stat().st_size,'tail':raw.decode(errors='replace')}
r=subprocess.run(['docker','inspect','glm53-sglang-shared-long-semantic-v4-20261017-43'],capture_output=True,text=True,timeout=15)
if r.returncode==0:
 d=json.loads(r.stdout)[0];assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1740-sglang-shared-long-semantic-v4'
 assert d['Image']=='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
 out['CPUclient']={'id':d['Id'],'state':d['State']}
else:out['CPUclient_inspect_rc']=r.returncode
print('GLM53_LONG_CHECKPOINT='+json.dumps(out),flush=True)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[43];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.43'
 code=re.sub(r'\bHOSTNAME\b',lambda _:repr(host),CODE)
 r=b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=90)
 (RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0},indent=2)+'\n')
 print('checkpoint',r['rc'],flush=True)
if __name__=='__main__':main()
