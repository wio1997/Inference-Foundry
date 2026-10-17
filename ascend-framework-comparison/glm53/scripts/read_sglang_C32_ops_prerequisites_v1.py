"""One-shot source-only eligibility capture from owned D36; no imports or NPU calls."""
import ast, hashlib, importlib.util, json, re, shlex
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1]
RID='20261017T2062-sglang-C32-ops-prerequisites-v1';RUN=ROOT/'runs'/RID
RUNTIME='config/C32-only-L04-restore/runtime.json'
RUNTIME_SHA='1100bc9f5c4a938becd0233c00d513e2e1a3384adc95ea18f257f30f215fd526'
def load(name,file,sha):
 p=P.with_name(file);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('inventory','inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
PROBE=r'''import pathlib,json,hashlib,sys,re
site=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
sg=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt')
paths={sg/'distributed/parallel_state.py',sg/'hardware_backend/npu/utils.py',sg/'layers/moe/token_dispatcher/deepep.py',site/'deep_ep/buffer.py',site/'deep_ep/ep_strategy.py',site/'deep_ep/strategies/low_latency_strategy.py',site/'torch_npu/distributed/distributed_c10d.py'}
for root in [site/'torch_npu',pathlib.Path('/usr/local/Ascend/ascend-toolkit/latest/opp')]:
 assert root.is_dir(),str(root)
 hits=[]
 for path in root.rglob('*'):
  if not path.is_file() or path.is_symlink() or path.stat().st_size>2*1024**2:continue
  norm=re.sub('[^a-z0-9]','',path.name.lower())
  if ('moedistributedispatchv2' in norm or 'moedistributecombinev2' in norm) and path.suffix in ['.json','.ini','.h','.py','.yaml']:hits.append(path)
 assert len(hits)<=96,(str(root),len(hits));paths.update(hits)
files=[];total=0
for path in sorted(paths):
 if not path.is_file():
  files.append({'path':str(path),'missing':True});continue
 raw=path.read_bytes();assert len(raw)<=2*1024**2;total+=len(raw);assert total<=16*1024**2
 files.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})
assert not any(n.startswith(('torch','sglang','deep_ep','memcache_hybrid','memfabric_hybrid')) for n in sys.modules)
print(json.dumps({'project_id':'glm53','files':files,'source_bytes':total,'framework_imports':False,'model_requests':0,'SDK_init_calls':0,'NPU_calls':0}))
'''
HOST=r'''import pathlib,subprocess,json,hashlib,gzip,base64,socket
assert socket.gethostname()==HOSTNAME
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0];assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED_RID
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED_RID
state=json.loads((root/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and state['model_started'] and 'model_exit_code' not in state
r=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-'],input=PROBE,capture_output=True,text=True,timeout=180);assert r.returncode==0,r.stderr
out=json.loads(r.stdout);out.update({'node':36,'container_id':CID,'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'stderr':r.stderr})
rows=subprocess.run(['docker','top',CID,'-eo','pid'],capture_output=True,text=True,timeout=20);assert rows.returncode==0;parents=[]
for line in rows.stdout.splitlines()[1:]:
 if not line.strip().isdigit():continue
 pid=int(line.strip());proc=pathlib.Path('/proc')/str(pid)
 try:namespace=int(next(x.split()[-1] for x in (proc/'status').read_text().splitlines() if x.startswith('NSpid:')))
 except (OSError,StopIteration):continue
 if namespace==state['model_pid']:
  assert CID in (proc/'cgroup').read_text();environment=dict(item.split(b'=',1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
  parents.append({'host_pid':pid,'namespace_pid':namespace,'allowed_environment':{name:environment.get(name.encode(),b'').decode() for name in ['HCCL_BUFFSIZE','HCCL_INTRA_PCIE_ENABLE','HCCL_INTRA_ROCE_ENABLE','DEEP_USE_MODE','MOE_SHARED_EXPERT_RANK_NUM','MOE_EXPERT_TOKEN_NUMS_TYPE','SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK','SGLANG_ZBAL_LOCAL_MEM_SIZE']}})
assert len(parents)==1;out['actual_model_allowed_environment']=parents[0]
archive=pathlib.Path(ARCHIVE);assert not archive.exists();archive.mkdir(parents=True)
'''
def render(code,values):
 pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b'
 code=re.sub(pattern,lambda m:repr(values[m[0]]),code);ast.parse(code);return code
def main():
 raw=(ROOT/RUNTIME).read_bytes();assert hashlib.sha256(raw).hexdigest()==RUNTIME_SHA;runtime=json.loads(raw);item=runtime['runtime_containers']['36']
 prior=json.loads((ROOT/'runs/20261017T2060-sglang-C32-D-two-step-native-v1/receipt.json').read_text());assert len(prior['nodes'])==4 and all(v==0 for v in prior['nodes'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'runtime_sha256':RUNTIME_SHA}
 try:
  pod,host=b.NODES[36];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.36'
  code=render(HOST+codec.ENCODER,{'HOSTNAME':host,'CID':item['container_id'],'NAME':item['name'],'IMAGE':runtime['model_image'],'SHARED_RID':runtime['shared_run_id'],'PROBE':PROBE,'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+RID})
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  result=codec.decode(b.invoke([command],code,timeout=240));(RUN/'node-36.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes']['36']=result['rc'];assert result['rc']==0
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
