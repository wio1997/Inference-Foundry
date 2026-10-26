#!/usr/bin/env python3
"""Derived probe: fix oversized quant summary; inspect offline image sources."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
prior=HERE/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
spec=importlib.util.spec_from_file_location('frozen_inventory',prior)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.RUN=base.ROOT/'runs'/'20261017T0205-inventory-compact'
base.HOST=base.HOST.replace("entry['scalar_fields']={k:v for k,v in q.items() if isinstance(v,(str,int,float,bool)) or v is None}",
    "from collections import Counter; entry['scalar_value_counts']=dict(Counter(str(v) for v in q.values() if isinstance(v,(str,int,float,bool)) or v is None)); entry['entry_count']=len(q)")
base.HOST=base.HOST.replace("for cid in containers.get('stdout','').splitlines()[:40]:", "for cid in []:")
base.HOST=base.HOST.replace("'processes']=cmd(['ps','-eo','pid,ppid,user,comm,etimes'])", "'processes']=cmd(['sh','-c',\"ps -eo pid,ppid,user,comm,etimes | grep -Ei 'PID|python|vllm|sglang|memcache|transfer|hixl|mooncake|llm|aisbench'\"])")
base.HOST=base.HOST.replace("print(json.dumps(r,ensure_ascii=False))",r'''
r['driver_version']=cmd(['cat','/usr/local/Ascend/driver/version.info'])
r['image_layout']=cmd(['docker','image','inspect',image,'--format','{{json .Config.WorkingDir}} {{json .GraphDriver}}'])
r['framework_dirs']=cmd(['find','/data/wio/ascend-framework-comparison','-maxdepth','3','-type','d'],15)
print(json.dumps(r,ensure_ascii=False))
''')

IMAGE_CODE=r'''
import pathlib,json,hashlib,os
roots=[pathlib.Path(p) for p in ['/vllm-workspace','/workspace','/sgl-workspace','/app','/opt','/usr/local/lib','/usr/local/Ascend'] if pathlib.Path(p).exists()]
out={'cwd':os.getcwd(),'roots':{},'sources':[],'matched_paths':[],'package_metadata':[]}
for root in roots: out['roots'][str(root)]=[p.name for p in root.iterdir()][:80]
budget=2200000
seen=set()
for root in roots:
 for parent,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if d not in ['.git','node_modules','tests','test','__pycache__','include','docs','examples','torch','triton','numpy','sympy','scipy','pandas','transformers','opencv_python.libs','tensorflow','rust','go']]
  for name in files:
   path=pathlib.Path(parent)/name; s=str(path)
   relevant=any(x in s.lower() for x in ['memfabric','memcache','mooncake','hixl','glm_moe_dsa','glm5','glm_5','ascend_transfer','ascend/conn','disaggregation/ascend','sfa_remote'])
   if name=='METADATA' and any(x in s.lower() for x in ['vllm','sglang','memfabric','memcache','mooncake','hixl','torch_npu']):
    out['package_metadata'].append({'path':s,'header':path.read_text(errors='replace')[:3000]})
   if not relevant: continue
   if len(out['matched_paths'])<1500: out['matched_paths'].append(s)
   if path.suffix not in ['.py','.json','.md','.txt','.conf','.h']: continue
   if name in seen or path.stat().st_size>200000 or budget<=0:continue
   # Preserve implementation, not arbitrary SDK documentation or generated headers.
   if '/site-packages/' in s or '/dist-packages/' in s or 'vllm_ascend' in s or '/sglang/' in s:
    data=path.read_bytes()
    if len(data)>budget:continue
    out['sources'].append({'path':s,'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')})
    seen.add(name);budget-=len(data)
print(json.dumps(out))
'''

def main():
 base.main()
 # base.main checks all eight live mappings and hostnames first. No NPU mounts.
 for n in (30,34):
  record=json.loads((base.RUN/('node-%s.json'%n)).read_text())
  h=json.loads(record['probe']['stdout'])
  image=('sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' if n==30 else 'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')
  assert h['image']['rc']==0 and h['image']['stdout'].startswith(image+' ')
  pod=base.NODES[n][0]
  import shlex
  # Owned ephemeral diagnostic container. No host mounts, devices, network, or service.
  hostcode='import subprocess,sys\np=subprocess.run('+repr(['docker','run','--rm','--pull=never','--read-only','--network=none','--pids-limit=128','--memory=2g','--cpus=2','--label','project_id=glm53','--name','glm53-source-inspect-20261017-'+str(n),'--entrypoint','python3',image,'-c',IMAGE_CODE])+',text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)\nprint(p.stdout,end="");print(p.stderr,file=sys.stderr,end="");sys.exit(p.returncode)\n'
  cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  result=base.invoke([cmd],hostcode,timeout=220)
  (base.RUN/('image-source-%s.json'%n)).write_text(json.dumps(result,indent=2))
  print(json.dumps({'image_source_node':n,'rc':result['rc'],'bytes':len(result['stdout'])}),flush=True)
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'model_requests':0,'npu_compute_requests':0,'ephemeral_diagnostic_containers':[30,34]}
 (base.RUN/'derived-receipt.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
