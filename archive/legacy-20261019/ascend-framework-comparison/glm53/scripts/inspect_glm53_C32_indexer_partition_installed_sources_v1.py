"""Owned installed-source audit after interrupted client exit; no framework imports."""
from pathlib import Path
import ast,base64,gzip,hashlib,importlib.util,json,os,shlex
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L11-indexer-partition-installed-source-v1'
RECOVERY='runs/20261018T-L09-interrupted-control-recovery-v2'
PROBE=r'''import pathlib,json,hashlib,gzip,base64,subprocess,sys
roots=ROOTS
paths=PATHS
out={'project_id':'glm53','framework':FRAMEWORK,'files':[],'missing':[],'repository_HEAD':{},'SDK_imported':False,'SDK_init_calls':0,'model_requests':0};total=0
for root in roots:
 p=pathlib.Path(root)
 if (p/'.git').exists():
  r=subprocess.run(['git','-C',root,'rev-parse','HEAD'],capture_output=True,text=True,timeout=15);out['repository_HEAD'][root]={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
for root in roots:
 for p in pathlib.Path(root).rglob('*.py'):
  if any(k in p.name for k in ['indexer','sfa']) and ('model_executor' in str(p) or 'vllm_ascend' in str(p) or 'sglang/srt/layers/attention' in str(p)):paths.append(str(p))
paths=sorted(set(paths));assert len(paths)<=96
for value in paths:
 p=pathlib.Path(value)
 if not p.is_file():out['missing'].append(value);continue
 assert not p.is_symlink() and any(p.resolve().is_relative_to(pathlib.Path(r).resolve()) for r in roots)
 raw=p.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=16*1024**2
 out['files'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})
assert not any(x.startswith(('torch','vllm','sglang','memcache_hybrid','memfabric_hybrid')) for x in sys.modules)
out['source_bytes']=total
raw=json.dumps(out).encode();print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'decoded_bytes':len(raw),'gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()}))
'''
HOST=r'''import pathlib,socket,subprocess,json,hashlib
assert socket.gethostname()==HOSTNAME
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=30);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED_RID
state=json.loads((pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED_RID/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and state['model_started'] and 'model_exit_code' not in state
r=subprocess.run(['docker','top',CID,'-eo','pid'],capture_output=True,text=True,timeout=20);assert r.returncode==0;parents=[]
for line in r.stdout.splitlines()[1:]:
 if not line.strip().isdigit():continue
 pid=int(line.strip());p=pathlib.Path('/proc')/str(pid)
 try:nspid=int(next(v.split()[-1] for v in (p/'status').read_text().splitlines() if v.startswith('NSpid:')))
 except (OSError,StopIteration):continue
 if nspid!=state['model_pid']:continue
 assert CID in (p/'cgroup').read_text();env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
 parents.append({'host_pid':pid,'namespace_pid':nspid,'argv':(p/'cmdline').read_bytes().decode().split('\0'),'selected_environment':{k:env.get(k.encode(),b'').decode() for k in ['DEEP_USE_MODE','HCCL_BUFFSIZE','DEEPEP_HCCL_BUFFSIZE','TASK_QUEUE_ENABLE','SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK']}})
assert len(parents)==1
r=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-'],input=PROBE,capture_output=True,text=True,timeout=120)
print(json.dumps({'project_id':'glm53','node':NODE,'container_id':CID,'image_id':IMAGE,'shared_run_id':SHARED_RID,'model_parent':parents[0],'source_probe':{'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr},'model_requests':0,'SDK_init_calls':0,'remote_mutations':False}))
'''
def render(src,values):
 import re
 src=re.sub(r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b',lambda m:repr(values[m[0]]),src);ast.parse(src);return src
def plans():
 sg=json.loads((ROOT/'config/C32-target-graph-runtime-v3/formal-C32-target-graph-v1.json').read_text());assert sg['native_qualification_sha256']=='bbb3eb81390e4aeca548fe6c055379c256a6122c3b73f2adcf4e8be5b1e3d52f'
 v=json.loads((ROOT/'results/vllm-formal-C32-v2-native-v2.json').read_text());assert v['pass'];vc=json.loads((ROOT/'config/vllm-correctness-v10/node-43.json').read_text())
 vr='/vllm-workspace/vllm';ar='/vllm-workspace/vllm-ascend';sr='/sgl-workspace/sglang/python/sglang'
 vp=[vr+'/vllm/'+p for p in ['model_executor/models/glm_moe_dsa.py','model_executor/models/deepseek_v2.py','model_executor/models/deepseek_v32.py']]
 sp=[sr+'/srt/'+p for p in ['models/deepseek_v2.py','models/registry.py','model_loader/utils.py','configs/model_config.py','layers/communicator.py','layers/dp_attention.py','managers/tp_worker.py','arg_groups/parallel_hook.py','arg_groups/scheduling_hook.py','hardware_backend/npu/memory_pool_npu.py']]
 return [('vllm',43,{'container_id':v['runtime_CIDs']['43'],'name':'glm53-vllm-shared-v4-20261017-43'},vc['image_id'],'20261017T1766-vllm-shared-pool-v4',[vr,ar],vp),('sglang',34,sg['runtime_containers']['34'],sg['model_image'],sg['shared_run_id'],['/sgl-workspace/sglang'],sp)]
def main():
 proof=json.loads((ROOT/'results/sglang-C32-L10-target-graph-v1-measurement-v1.json').read_text());assert proof['measurement_valid'] is True
 rpc=json.loads((ROOT/'runs/20261018T-L10-sglang-formal-C32-target-graph-v1/controller-rpc-state.json').read_text());assert rpc['done_received'] and rpc['connection_rc']==0
 p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777';spec=importlib.util.spec_from_file_location('inventory',p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','nodes':{},'model_requests':0,'SDK_init_calls':0,'remote_mutations':False}
 try:
  for fw,node,item,image,shared,roots,paths in plans():
   probe=render(PROBE,{'ROOTS':roots,'PATHS':paths,'FRAMEWORK':fw});code=render(HOST,{'HOSTNAME':b.NODES[node][1],'CID':item['container_id'],'NAME':item['name'],'IMAGE':image,'SHARED_RID':shared,'PROBE':probe,'NODE':node});(run/(fw+'-host-source.py')).write_text(code)
   pod=b.NODES[node][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);(run/(fw+'-mapping.json')).write_text(json.dumps(mapped,indent=2)+'\n');assert mapped['rc']==0 and mapped['stdout']=='172.16.12.'+str(node)
   command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);out=b.invoke([command],code,timeout=180);(run/(fw+'.json')).write_text(json.dumps(out,indent=2)+'\n');receipt['nodes'][fw]=out['rc'];assert out['rc']==0
   d=json.loads(out['stdout']);assert d['source_probe']['rc']==0,d['source_probe']['stderr'];wire=json.loads(d['source_probe']['stdout']);raw=gzip.decompress(base64.b64decode(wire['gzip_base64'],validate=True));assert len(raw)==wire['decoded_bytes'] and hashlib.sha256(raw).hexdigest()==wire['sha256'];source=json.loads(raw);(run/(fw+'-source.json')).write_bytes(raw)
   dest=run/'source'/fw;dest.mkdir(parents=True)
   for x in source['files']:
    raw=x['source'].encode();assert hashlib.sha256(raw).hexdigest()==x['sha256'];(dest/(x['path'].strip('/').replace('/','__'))).write_bytes(raw)
   print(json.dumps({'framework':fw,'node':node,'source_files':len(source['files']),'source_bytes':source['source_bytes'],'missing_files':source['missing'],'model_requests':0}),flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
