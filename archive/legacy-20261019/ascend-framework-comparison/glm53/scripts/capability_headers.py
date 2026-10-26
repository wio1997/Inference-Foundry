#!/usr/bin/env python3
"""Bounded offline SDK/source and tensor-header evidence; ZCode-only execution."""
import hashlib, importlib.util, json, shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent
prior=HERE/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
RUN=base.ROOT/'runs'/'20261017T0210-capability-headers'
CODE=r'''
import sys,os,pathlib,json,hashlib,importlib.metadata as md
out={'python':sys.version,'sys_path':sys.path,'distributions':[],'sources':[],'libraries':[]}
roots=[]
for d in md.distributions():
 name=d.metadata.get('Name','')
 if any(x in name.lower() for x in ['vllm','sglang','memfabric','memcache','mooncake','torch-npu','torch_npu','hixl']):
  out['distributions'].append({'name':name,'version':d.version,'root':str(d.locate_file(''))})
  roots.append(pathlib.Path(d.locate_file('')))
roots+= [pathlib.Path(p) for p in ['/vllm-workspace/vllm','/vllm-workspace/vllm-ascend','/sgl-workspace/sglang/python'] if pathlib.Path(p).exists()]
budget=3000000;seen=set()
exact={'server_args.py','parallel.py','speculative.py','kv_cache_interface.py','glm_moe_dsa.py','glm_moe_dsa_nextn.py','glm4_moe.py','glm4_moe_lite.py','glm4_moe_lite_nextn.py','glm_moe_dsa_mtp.py','sfa.py','sfa_v1.py','sfa_remote_d2h_connector.py'}
for root in set(roots):
 for parent,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if d not in ['.git','node_modules','__pycache__','tests','test','torch','numpy','sympy','scipy','transformers','pandas','glm5next','glm5_next','include','docs','examples']]
  for name in files:
   path=pathlib.Path(parent)/name;p=str(path)
   sdk=any(x in p.lower() for x in ['memfabric_hybrid','memcache_hybrid','mooncake/engine'])
   if sdk and ('.so' in name or path.suffix in ['.conf','.ini']):
    out['libraries'].append({'path':p,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest() if '.so' in name and path.stat().st_size<300000000 else None})
   relevant=(sdk and path.suffix in ['.py','.conf','.ini']) or name in exact or ('sfa' in name and path.suffix=='.py') or (name=='__init__.py' and 'kv_transfer' in p)
   if not relevant or p in seen or path.stat().st_size>350000:continue
   data=path.read_bytes()
   if len(data)>budget:continue
   seen.add(p);budget-=len(data)
   out['sources'].append({'path':p,'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode(errors='replace')})
print(json.dumps(out))
'''
HOST=r'''
import pathlib,json,struct,collections,re,hashlib,subprocess,socket,sys
assert socket.gethostname()==sys.argv[1]
m=pathlib.Path('/data/wio/models/GLM-5.3-w8a8')
out={'hostname':socket.gethostname(),'groups':{},'patterns':{},'header_bytes':0,'files':[],'samples':[]}
for p in sorted(m.glob('*.safetensors')):
 with p.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]
  assert n<16000000 and out['header_bytes']+n<128000000
  data=f.read(n);q=json.loads(data)
 out['header_bytes']+=n
 out['files'].append({'name':p.name,'header_bytes':n,'header_sha256':hashlib.sha256(data).hexdigest()})
 for k,v in q.items():
  if k=='__metadata__':continue
  size=v['data_offsets'][1]-v['data_offsets'][0]
  layer=re.search(r'layers\.(\d+)\.',k); draft=bool(layer and int(layer.group(1))>=78)
  kind=('draft_' if draft else 'target_')+('routed' if re.search(r'\.experts\.\d+\.',k) else 'shared' if '.shared_experts.' in k else 'dense')
  group=out['groups'].setdefault(kind,{'bytes':0,'tensors':0,'dtypes':{}});group['bytes']+=size;group['tensors']+=1
  group['dtypes'][v['dtype']]=group['dtypes'].get(v['dtype'],0)+size
  pattern=re.sub(r'layers\.\d+','layers.L',re.sub(r'experts\.\d+','experts.E',k))
  g=out['patterns'].setdefault(pattern,{'bytes':0,'tensors':0,'example_shape':v['shape'],'dtype':v['dtype']});g['bytes']+=size;g['tensors']+=1
  if len(out['samples'])<40:out['samples'].append({'name':k,'shape':v['shape'],'dtype':v['dtype']})
out['total_tensor_bytes']=sum(g['bytes'] for g in out['groups'].values())
out['process_status']=subprocess.run(['sh','-c',"ps -eo pid,ppid,stat,comm | grep -Ei 'PID|python|vllm|sglang|memcache|mooncake'"],capture_output=True,text=True,timeout=10).stdout
print(json.dumps(out))
'''
def host(n,code,args=(),timeout=240):
 pod,hostname=base.NODES[n]
 mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(n)
 cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-',*args])
 return base.invoke([cmd],code,timeout=timeout)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 r=host(30,HOST,[base.NODES[30][1]],timeout=180)
 (RUN/'tensor-headers-30.json').write_text(json.dumps(r,indent=2))
 for n,image in [(30,'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'),(34,'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')]:
  args=['docker','run','--rm','--pull=never','--read-only','--network=none','--pids-limit=128','--memory=3g','--cpus=2','--label','project_id=glm53','--name','glm53-capability-20261017-'+str(n),'--entrypoint','python3',image,'-c',CODE]
  hc='import subprocess,socket,sys\nassert socket.gethostname()=='+repr(base.NODES[n][1])+'\nr=subprocess.run('+repr(args)+',text=True,capture_output=True,timeout=180)\nprint(r.stdout,end="");print(r.stderr,file=sys.stderr,end="");sys.exit(r.returncode)\n'
  r=host(n,hc);(RUN/('sdk-source-%s.json'%n)).write_text(json.dumps(r,indent=2));print(n,r['rc'],len(r['stdout']),flush=True)
 # Only read status of the pre-existing CPU AISBench processes; do not signal them.
 r=host(43,'import subprocess,json,socket\nassert socket.gethostname()=='+repr(base.NODES[43][1])+'\nprint(subprocess.run(["ps","-eo","pid,ppid,stat,comm"],capture_output=True,text=True,timeout=10).stdout)\n')
 (RUN/'process-status-43.json').write_text(json.dumps(r,indent=2))
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_requests':0,'npu_compute_requests':0,'diagnostic_containers':[30,34],'header_payload_read':False},indent=2))
if __name__=='__main__':main()
