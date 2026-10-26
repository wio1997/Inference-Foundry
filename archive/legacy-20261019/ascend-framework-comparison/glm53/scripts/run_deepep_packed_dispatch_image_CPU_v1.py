"""Once-only CPU property qualification in actual fixed model image; existing L04 unchanged."""
import ast,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L09-packed-dispatch-image-CPU-v1'
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',p);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
PATHS={'normal_strategy.py': 'runs/20261018T-L09-source-selection-v1/normal_strategy.py', 'glm53_deepep_metadata_CPU_fixture_v1.py': 'scripts/glm53_deepep_metadata_CPU_fixture_v1.py', 'glm53_deepep_metadata_batch_v1.py': 'scripts/glm53_deepep_metadata_batch_v1.py', 'glm53_deepep_packed_dispatch_v1.py': 'scripts/glm53_deepep_packed_dispatch_v1.py', 'glm53_deepep_packed_dispatch_CPU_fixture_v1.py': 'scripts/glm53_deepep_packed_dispatch_CPU_fixture_v1.py'}
FROZEN={'normal_strategy.py': '48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84', 'glm53_deepep_metadata_CPU_fixture_v1.py': 'b82d4b25c9931d4cfa364dbbc0a1214d87fc7564b1a8855f6abb4b51180a9133', 'glm53_deepep_metadata_batch_v1.py': 'daf4825ab2579af4cca617cf2746b8a97d30020e7abde47277bc4abe7e009363', 'glm53_deepep_packed_dispatch_v1.py': '336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb', 'glm53_deepep_packed_dispatch_CPU_fixture_v1.py': '0539d58d6230d750ab997ad73b25046fcc277c69276214a00838b9dfd32a944b'}
FILES={k:(ROOT/v).read_text() for k,v in PATHS.items()}
for k,v in FILES.items():assert hashlib.sha256(v.encode()).hexdigest()==FROZEN[k]
p=ROOT/'config/C32-window-L04-rollback-runtime-v2/runtime.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='5ccd3842d3cdba4df2d104f00666ee7f4eea683ad9f721e30cf405268b54fa00';RUNTIME=json.loads(p.read_text())
HOST=r'''import pathlib,json,hashlib,subprocess,socket,urllib.request,time
assert socket.gethostname()==HOSTNAME
def cmd(args,t=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=t);assert r.returncode==0,(args,r.returncode,r.stderr);return r.stdout
assert cmd(['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip()==IMAGE
shared=pathlib.Path(SHARED)
def snapshot():
 d=json.loads(cmd(['docker','inspect',CID]))[0]
 assert d['Id']==CID and d['Name']=='/'+MODEL_NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==MODEL_RID
 supervisor=json.loads((shared/'supervisor-status.json').read_text());assert supervisor['phase']=='MODEL_RUNNING' and 'model_exit_code' not in supervisor
 npu=cmd(['npu-smi','info']);pids=[]
 for line in npu.split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
   pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();pids.append(pid)
 assert len(pids)==18
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));health={}
 for endpoint in ['health','metrics']:
  with opener.open('http://172.16.12.36:30600/'+endpoint,timeout=15) as res:
   raw=res.read(16*1024**2+1);assert res.status==200 and len(raw)<=16*1024**2
   health[endpoint]={'status':200,'sha256':hashlib.sha256(raw).hexdigest()}
   if endpoint=='metrics':
    gauges=[l for l in raw.decode().splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(float(l.rsplit(' ',1)[1])==0 for l in gauges);health[endpoint]['idle_gauges']=gauges
 return {'model_container_id':CID,'supervisor':supervisor,'owned_NPU_context_PIDs':sorted(pids),'health':health,'npu_sha256':hashlib.sha256(npu.encode()).hexdigest()}
before=snapshot();root=pathlib.Path(REMOTE);root.mkdir(exist_ok=False)
for name,source in FILES.items():
 assert pathlib.PurePosixPath(name).name==name and hashlib.sha256(source.encode()).hexdigest()==FROZEN[name]
 (root/name).write_text(source)
args=['docker','run','--rm','--pull=never','--network=none','--pids-limit=512','--memory=8g','--cpus=2','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-L09-packed-dispatch-CPU-v1-20261018-36','--workdir','/run/glm53','-v',REMOTE+':/run/glm53','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONNOUSERSITE=1','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','CUDA_VISIBLE_DEVICES=','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HOME=/run/glm53/home','-e','XDG_CACHE_HOME=/run/glm53/cache','--entrypoint','timeout',IMAGE,'300','python3','/run/glm53/glm53_deepep_packed_dispatch_CPU_fixture_v1.py']
result=subprocess.run(args,capture_output=True,text=True,timeout=350)
(root/'fixture.stdout.txt').write_text(result.stdout);(root/'fixture.stderr.txt').write_text(result.stderr)
after=snapshot();assert before['owned_NPU_context_PIDs']==after['owned_NPU_context_PIDs']
out={'project_id':'glm53','rc':result.returncode,'before':before,'after':after,'model_requests':0,'SDK_init_calls':0,'NPU_calls_by_fixture':0,'NPU_devices_mapped':False,'model_service_control_calls':0,'image':IMAGE,'files_sha256':FROZEN,'stdout_bytes':len(result.stdout.encode()),'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),'stderr':result.stderr,'stdout_tail':result.stdout[-12000:]}
if (root/'fixture-result.json').is_file():out['fixture_result']=json.loads((root/'fixture-result.json').read_text())
print(json.dumps(out));raise SystemExit(result.returncode)
'''
def render():
 v=RUNTIME['runtime_containers']['36'];values={'HOSTNAME':base.NODES[36][1],'CID':v['container_id'],'MODEL_NAME':v['name'],'MODEL_RID':RUNTIME['shared_run_id'],'SHARED':RUNTIME['shared_remote_root'],'IMAGE':RUNTIME['model_image'],'RID':RID,'REMOTE':'/data/wio/inference-foundry/glm53/runs/'+RID,'FILES':FILES,'FROZEN':FROZEN}
 code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST);ast.parse(code);return code
def main():
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,_=base.NODES[36];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);(run/'mapping-36.json').write_text(json.dumps(mapping,indent=2)+'\n');assert mapping['rc']==0 and mapping['stdout']=='172.16.12.36'
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);result=base.invoke([command],render(),timeout=450);(run/'node-36.json').write_text(json.dumps(result,indent=2)+'\n')
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files_sha256':FROZEN,'rc':result['rc'],'model_requests':0,'SDK_init_calls':0,'model_service_control_calls':0,'max_SSH':1};(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('fixture',result['rc'],flush=True)
if __name__=='__main__':main()
