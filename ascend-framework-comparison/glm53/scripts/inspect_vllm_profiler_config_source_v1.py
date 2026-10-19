#!/usr/bin/env python3
"""CPU-only exact profiler/pool allocation source; no framework import/NPU devices."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1482-vllm-profiler-config-source-v1'
IMAGES={30:'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'}
CODE="import pathlib,json,hashlib\nroots=[pathlib.Path('/vllm-workspace'),pathlib.Path('/usr/local/lib/python3.11/site-packages/vllm'),pathlib.Path('/usr/local/lib/python3.12/site-packages/vllm')];paths=[]\nfor root in roots:\n if root.exists():\n  for p in root.rglob('profiler.py'):\n   if 'class ProfilerConfig' in p.read_text(errors='replace'):paths.append(p)\n  for p in root.rglob('arg_utils.py'):\n   if '/vllm/' in str(p):paths.append(p)\nfiles=[]\nfor p in dict.fromkeys(paths):\n raw=p.read_bytes();assert len(raw)<500000;files.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()})\nassert any('class ProfilerConfig' in f['text'] for f in files)\nprint('GLM53_NATIVE_PROFILER_CONFIG_SOURCE='+json.dumps({'files':files,'model_requests':0,'npu_allocations':0}))\n"
def capture(node):
 pod,host=b.NODES[node];image=IMAGES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']==f'172.16.12.{node}'
 code=re.sub(r'\bNODE\b',repr(node),CODE)
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=2g','--pids-limit=128','--label','project_id=glm53','--name',f'glm53-profiler-config-source-v1-20261017-{node}','--entrypoint','python3',image,'-c',code]
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
