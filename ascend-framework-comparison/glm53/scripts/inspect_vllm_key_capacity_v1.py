#!/usr/bin/env python3
"""Metadata-only, existing known semantic prefix keys across head ranks; no BM/NPU/model requests."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1390-vllm-key-capacity-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1390-vllm-key-capacity-v1'
CODE=(P.parent/'glm53_vllm_key_capacity_probe_v1.py').read_text();assert hashlib.sha256(CODE.encode()).hexdigest()=='2270e8a263b330889551bf63709db63361c2921ebf3dab1e6a2440a85240dd91'
CID='101a44f8ff8913bc5d103fb28ac3c09754005652ef06f805195187dd17db8a3c'
HOST=r'''import pathlib,socket,subprocess,json,hashlib,sys
assert socket.gethostname()==HOSTNAME
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay';root.mkdir(parents=True)
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/glm53-vllm-correctness-v4-20261017-43' and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53'
data=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1');assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()=='ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece'
request={'project_id':'glm53','cases':[]}
for i in [0,1]:
 c=json.loads((data/f'semantic-{i}.json').read_text());request['cases'].append({'namespace':f'glm53-sem-vllm-v4-20261017-case{i}-warm','prefix_ids':c['prefix_ids'],'prefix_sha256_le_u32':c['prefix_sha256_le_u32']})
(root/'probe.py').write_text(CODE)
boot='export PYTHONPATH=/opt/glm53/sdk13:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; export PYTHONDONTWRITEBYTECODE=1; exec python3 -c "$1"'
r=subprocess.run(['docker','exec','-i',CID,'bash','-c',boot,'glm53-key-capacity',CODE],input=json.dumps(request),capture_output=True,text=True,timeout=180)
(root/'stdout.txt').write_text(r.stdout);(root/'stderr.txt').write_text(r.stderr)
lines=[l for l in r.stdout.splitlines() if l.startswith('GLM53_KEY_CAPACITY_METADATA=')];report=json.loads(lines[0].split('=',1)[1]) if len(lines)==1 else None
print('GLM53_KEY_CAPACITY_AUDIT='+json.dumps({'rc':r.returncode,'report':report,'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr.encode()).hexdigest(),'stderr_tail':r.stderr[-4000:]}),flush=True)
sys.exit(r.returncode)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[43];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']=='172.16.12.43'
 code=HOST
 for k,v in [('HOSTNAME',host),('REMOTE',REMOTE),('CID',CID),('CODE',CODE)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 r=b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=240)
 (RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'init_bm':False,'NPU_allocations':0},indent=2)+'\n');print('capacity',r['rc'],flush=True)
if __name__=='__main__':main()
