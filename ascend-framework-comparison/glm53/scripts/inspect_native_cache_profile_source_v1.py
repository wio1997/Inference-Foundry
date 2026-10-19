#!/usr/bin/env python3
"""CPU-only exact profiler/pool allocation source; no framework import/NPU devices."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1626-native-cache-profile-source-v1'
IMAGES={30:'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83',34:'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'}
CODE="import pathlib,json,hashlib\nfiles=[];listing=[];budget=3500000\nroots=[pathlib.Path('/vllm-workspace/vllm'),pathlib.Path('/sgl-workspace/sglang/python/sglang/srt')]\nterms=['enable_prompt_tokens_details','enable-prompt-tokens-details','prompt_tokens_details','class Profiler','def start_profile','def stop_profile','torch.profiler.profile']\nfor root in roots:\n if not root.is_dir():continue\n for p in root.rglob('*.py'):\n  if not any(x in str(p) for x in ['/entrypoints/','/profiler/','/managers/','/profiler.py','/arg_groups/']):continue\n  raw=p.read_bytes()\n  if len(raw)>700000:continue\n  txt=raw.decode(errors='replace');hits=[{'line':i+1,'text':line[:350]} for i,line in enumerate(txt.splitlines()) if any(t in line for t in terms)]\n  if not hits:continue\n  listing.append({'path':str(p),'hits':hits})\n  if len(raw)<=budget:\n   budget-=len(raw);files.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':txt})\nprint('GLM53_NATIVE_CACHE_PROFILE_SOURCE='+json.dumps({'files':files,'listing':listing,'model_requests':0,'npu_allocations':0}))\n"

def capture(node):
 pod,host=b.NODES[node];image=IMAGES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']==f'172.16.12.{node}'
 code=re.sub(r'\bNODE\b',repr(node),CODE)
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=2g','--pids-limit=128','--label','project_id=glm53','--name',f'glm53-cache-profile-source-v1-20261017-{node}','--entrypoint','python3',image,'-c',code]
 hostcode=r'''import socket,subprocess,sys
assert socket.gethostname()==HOST
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert r.returncode==0 and r.stdout.strip()==IMAGE
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
