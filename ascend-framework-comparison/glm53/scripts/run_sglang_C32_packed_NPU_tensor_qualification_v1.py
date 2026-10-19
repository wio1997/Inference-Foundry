"""One bounded synthetic tensor probe beside idle currentL04; no service/SDK calls."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re,shlex
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L09-packed-dispatch-NPU-tensor-v1'
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',p);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
p=ROOT/'config/C32-window-L04-rollback-runtime-v2/runtime.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='5ccd3842d3cdba4df2d104f00666ee7f4eea683ad9f721e30cf405268b54fa00';runtime=json.loads(p.read_text())
p=ROOT/'scripts/glm53_deepep_packed_dispatch_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb';PACK=p.read_text()
p=ROOT/'scripts/glm53_packed_dispatch_NPU_tensor_fixture_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='59998e3d90514cd4712fe5d5ff0d941b421e95dcd83fc6e624eecdf880245820';INNER='PACK_SOURCE='+repr(PACK)+'\n'+p.read_text();ast.parse(INNER)
HOST=r'''import pathlib,json,hashlib,subprocess,socket,urllib.request,re,time
assert socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);assert p.returncode==0,(p.returncode,p.stderr);return p.stdout
shared=pathlib.Path(SHARED)
def snapshot():
 d=json.loads(cmd(['docker','inspect',CID]))[0];assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==MODEL_RID and d['State']['Running']
 status=json.loads((shared/'supervisor-status.json').read_text());assert status['phase']=='MODEL_RUNNING' and 'model_exit_code' not in status
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));g=[]
 for endpoint in ['health','metrics']:
  with opener.open('http://172.16.12.36:30600/'+endpoint,timeout=15) as res:
   raw=res.read(16*1024**2+1);assert res.status==200 and len(raw)<=16*1024**2
   if endpoint=='metrics':g=[l for l in raw.decode().splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert g and all(float(l.rsplit(' ',1)[1])==0 for l in g)
 text=cmd(['npu-smi','info']);pids=[];physical0=None
 for line in text.splitlines():
  fields=[x.strip() for x in line.split('|')]
  if len(fields)>=5 and fields[1].split()==['0','0'] and ':' in fields[2]:
   pairs=re.findall(r'(\d+)\s*/\s*(\d+)',fields[-2]);assert pairs;used,total=map(int,pairs[-1]);physical0={'used_MiB':used,'total_MiB':total}
 for line in text.split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
   pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();pids.append(pid)
 assert len(pids)==18 and physical0
 return {'supervisor':status,'owned_model_context_pids':sorted(pids),'physical_device0':physical0,'idle_gauges':g,'npu_SMI_SHA256':hashlib.sha256(text.encode()).hexdigest()}
before=snapshot();free=before['physical_device0']['total_MiB']-before['physical_device0']['used_MiB'];assert free>=2048,'Insufficient spare HBM: preserve model and do not start probe'
root=pathlib.Path(REMOTE);root.mkdir(exist_ok=False)
args=['docker','exec',CID,'env','TORCH_DEVICE_BACKEND_AUTOLOAD=0','ASCEND_RT_VISIBLE_DEVICES=0','ASCEND_VISIBLE_DEVICES=0','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','HOME=/tmp/glm53-L09-tensor-home','XDG_CACHE_HOME=/tmp/glm53-L09-tensor-cache','ASCEND_WORK_PATH=/tmp/glm53-L09-tensor-work','ASCEND_CACHE_PATH=/tmp/glm53-L09-tensor-cache','timeout','--kill-after=10','120','python3','-c',INNER]
p=subprocess.run(args,capture_output=True,text=True,timeout=160)
(root/'fixture.stdout.txt').write_text(p.stdout);(root/'fixture.stderr.txt').write_text(p.stderr)
after=snapshot();assert before['owned_model_context_pids']==after['owned_model_context_pids'],'Probe context not released; preserve all resources for review'
out={'project_id':'glm53','container_id':CID,'image':IMAGE,'probe_rc':p.returncode,'before':before,'after':after,'physical_free_MiB_before':free,'external_model_HTTP_requests':0,'SDK_init_calls':0,'model_service_control_calls':0,'fixture_helper_SHA256':'336a64b8f6657541f24daa49588891072d350b4a0d9d94b383c6f3fe08cea5cb','stdout':p.stdout,'stderr':p.stderr,'peak_memory_limit_scope':'Only synthetic child Torch allocator fraction0.01; current model unchanged'}
if p.returncode==0:
 line=next(l for l in p.stdout.splitlines() if l.startswith('GLM53_PACKED_NPU_TENSOR_FIXTURE='));out['result']=json.loads(line.split('=',1)[1])
print(json.dumps(out));raise SystemExit(p.returncode)
'''
def render():
 v=runtime['runtime_containers']['36'];values={'HOSTNAME':base.NODES[36][1],'CID':v['container_id'],'NAME':v['name'],'IMAGE':runtime['model_image'],'MODEL_RID':runtime['shared_run_id'],'SHARED':runtime['shared_remote_root'],'REMOTE':'/data/wio/inference-foundry/glm53/runs/'+RID,'INNER':INNER};code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST);ast.parse(code);return code
def main():
 # No overlap with the temporal BEFORE performance window.
 measured=json.loads((ROOT/'results/sglang-C32-L09-L04-before-v1-measurement-v1.json').read_text());assert measured['measurement_valid'] and measured['run_id']=='20261018T-L09-sglang-formal-C32-L04-before-v1' and measured['runtime_CIDs']=={n:v['container_id'] for n,v in runtime['runtime_containers'].items()}
 rpc=json.loads((ROOT/'runs/20261018T-L09-sglang-formal-C32-L04-before-v1/controller-rpc-state.json').read_text());assert rpc['done_received']
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,_=base.NODES[36];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);(run/'mapping-36.json').write_text(json.dumps(mapping,indent=2)+'\n');assert mapping['rc']==0 and mapping['stdout']=='172.16.12.36'
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);result=base.invoke([command],render(),timeout=230);(run/'node-36.json').write_text(json.dumps(result,indent=2)+'\n');(run/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rc':result['rc'],'max_SSH':1,'external_model_HTTP_requests':0,'SDK_init_calls':0,'model_service_control_calls':0},indent=2)+'\n');print(result['rc'],flush=True)
if __name__=='__main__':main()
