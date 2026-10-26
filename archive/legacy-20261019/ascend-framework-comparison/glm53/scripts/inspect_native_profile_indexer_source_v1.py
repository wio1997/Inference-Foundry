#!/usr/bin/env python3
"""Read exact running-image native profiler/model/indexer allocation source only."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1240-native-profile-indexer-source-v1'
TARGETS={30:('c37c2e631f60ff4a4517d129f0a5341a56ab06c0d448463b466ca3c35bb06786','glm53-vllm-correctness-v4-20261017-30','sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'),34:('c2964d10eadaebf41d2d3f9817baa3a42373424848dcd44bd672fffe8c1ff7f2','glm53-sglang-correctness-v3-20261017-34','sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')}
CODE=r'''import pathlib,json,hashlib
if NODE==30:
 root=pathlib.Path('/vllm-workspace/vllm');a=pathlib.Path('/vllm-workspace/vllm-ascend');paths=[a/'vllm_ascend/worker/worker.py',root/'vllm/profiler/wrapper.py',root/'vllm/config/observability.py'];search=[root/'vllm/entrypoints']
else:
 root=pathlib.Path('/sgl-workspace/sglang/python/sglang');paths=[root/'srt/models/deepseek_v2.py',root/'srt/models/glm_moe_dsa.py',root/'srt/model_executor/model_runner.py'];search=[root/'srt/managers',root/'srt/layers/attention',root/'srt/models']
 for p in (root/'srt').rglob('*.py'):
  if 'profil' in p.name:paths.append(p)
files=[];budget=2900000
for p in paths:
 if not p.is_file():continue
 raw=p.read_bytes()
 if len(raw)>1000000 or len(raw)>budget:continue
 budget-=len(raw);files.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()})
snippets=[];budget=450000
for folder in search:
 for p in folder.rglob('*.py'):
  if p.stat().st_size>1200000:continue
  text=p.read_text();lines=text.splitlines();matches=[i for i,line in enumerate(lines) if any(x in line for x in (('start_profile','stop_profile') if NODE==30 else ('indexer_layer_ids','GlmMoeDsaForCausalLM','index_share_for','indexer_reuse','indexer_skip')))]
  if not matches:continue
  indices=sorted({j for i in matches for j in range(max(0,i-6),min(len(lines),i+12))});chosen=[[i+1,lines[i]] for i in indices];size=len(json.dumps(chosen))
  if size>budget:continue
  budget-=size;snippets.append({'path':str(p),'sha256':hashlib.sha256(text.encode()).hexdigest(),'lines':chosen})
meta={}
if NODE==34:
 p=pathlib.Path('/data/wio/models/GLM-5.3-w8a8/config.json');raw=p.read_bytes();d=json.loads(raw);meta={'config_sha256':hashlib.sha256(raw).hexdigest(),'index_nextn_layer_fields':{k:v for k,v in d.items() if any(x in k.lower() for x in ('index','nextn','layer'))}}
print('GLM53_NATIVE_PROFILE_INDEXER_SOURCE='+json.dumps({'files':files,'snippets':snippets,'metadata':meta}),flush=True)
'''
def inspect(node):
 pod,host=b.NODES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.'+str(node)
 cid,name,image=TARGETS[node];payload='NODE='+repr(node)+'\n'+CODE
 code=r'''import socket,subprocess,json,sys
assert socket.gethostname()==HOST
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=15);assert r.returncode==0;q=json.loads(r.stdout)[0]
assert q['Id']==CID and q['Name']=='/'+NAME and q['Image']==IMAGE and q['Config']['Labels']['project_id']=='glm53' and q['State']['Running']
r=subprocess.run(['docker','exec',CID,'python3','-c',PAYLOAD],capture_output=True,text=True,timeout=60);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
'''
 for k,v in [('HOST',host),('CID',cid),('NAME',name),('IMAGE',image),('PAYLOAD',payload)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=130)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'model_requests':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(inspect,n):n for n in TARGETS}
   for future in concurrent.futures.as_completed(fs):
    n=fs[future]
    try:r=future.result()
    except Exception as error:r={'error':repr(error)}
    (RUN/f'source-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][n]=r.get('rc',r.get('error'))
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
