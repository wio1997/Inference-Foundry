#!/usr/bin/env python3
"""Read exact B103 native API/PD routing/cachekey source; no imports/requests."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('b',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1000-sglang-pd-client-contract-v1';CID='37c64e8270d9adfd90de047538b5dff35192103371be068f33bea9bb42bbd452';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
CODE=r'''
import pathlib,hashlib,json,importlib.util
root=pathlib.Path('/sgl-workspace/sglang/python/sglang');paths=[]
for name in ['srt/entrypoints/openai/serving_completions.py','srt/entrypoints/openai/protocol.py','srt/entrypoints/http_server.py','srt/managers/io_struct.py','srt/disaggregation/mini_lb.py','srt/disaggregation/launch_lb.py','srt/mem_cache/radix_cache.py','srt/mem_cache/hiradix_cache.py','srt/managers/cache_controller.py','srt/mem_cache/storage/npu_memcache_store.py','srt/mem_cache/storage/npu_memcache.py','srt/utils/network.py']:paths.append(root/name)
listing={}
for folder in ['srt/disaggregation','srt/mem_cache/storage','srt/entrypoints/openai']:
 directory=root/folder;listing[str(directory)]=[str(p) for p in directory.glob('*.py')]
for package in ['sglang_router','sgl_model_gateway']:
 spec=importlib.util.find_spec(package)
 if spec and spec.origin:
  directory=pathlib.Path(spec.origin).parent;listing[package]=[str(p) for p in directory.rglob('*.py')]
  paths.extend(pathlib.Path(p) for p in listing[package] if any(x in p for x in ('launch_router','router.py','__init__')))
files=[];budget=3000000
for p in paths:
 if not p.is_file():continue
 data=p.read_bytes()
 if len(data)>650000 or len(data)>budget:continue
 budget-=len(data);files.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()})
print('GLM53_SGLANG_PD_SOURCE='+json.dumps({'files':files,'listing':listing}),flush=True)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[34];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.34'
 hostcode=r'''import socket,subprocess,json,sys
assert socket.gethostname()==HOST
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=15);assert r.returncode==0
q=json.loads(r.stdout)[0];assert q['Id']==CID and q['Name']=='/glm53-sglang-correctness-v1-20261017-34' and q['Image']==IMAGE and q['Config']['Labels']['project_id']=='glm53'
r=subprocess.run(['docker','exec',CID,'python3','-c',CODE],capture_output=True,text=True,timeout=60);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
'''
 for k,v in [('HOST',host),('CID',CID),('IMAGE',IMAGE),('CODE',CODE)]:hostcode=re.sub(r'\b'+k+r'\b',lambda _:repr(v),hostcode)
 r=b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],hostcode,timeout=120);(RUN/'source-34.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0},indent=2)+'\n');print('source',r['rc'],flush=True)
if __name__=='__main__':main()
