"""Owned installed-source audit after interrupted client exit; no framework imports."""
from pathlib import Path
import ast,base64,gzip,hashlib,importlib.util,json,os,shlex
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-C32-alignment-comm-source-v1'
RECOVERY='runs/20261018T-L09-interrupted-control-recovery-v2'
PROBE=r'''import pathlib,json,hashlib,gzip,base64,subprocess,sys
roots=ROOTS
paths=PATHS
out={'project_id':'glm53','framework':FRAMEWORK,'files':[],'missing':[],'repository_HEAD':{},'SDK_imported':False,'SDK_init_calls':0,'model_requests':0};total=0
for root in roots:
 p=pathlib.Path(root)
 if (p/'.git').exists():
  r=subprocess.run(['git','-C',root,'rev-parse','HEAD'],capture_output=True,text=True,timeout=15);out['repository_HEAD'][root]={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
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
 parents.append({'host_pid':pid,'namespace_pid':nspid,'argv':(p/'cmdline').read_bytes().decode().split('\0'),'selected_environment':{k:env.get(k.encode(),b'').decode() for k in ['DEEP_USE_MODE','HCCL_BUFFSIZE','DEEPEP_HCCL_BUFFSIZE','TASK_QUEUE_ENABLE','SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK','VLLM_ASCEND_ENABLE_FUSED_MC2','VLLM_ASCEND_ENABLE_PREFILL_MC2']}})
assert len(parents)==1
r=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-'],input=PROBE,capture_output=True,text=True,timeout=120)
print(json.dumps({'project_id':'glm53','node':NODE,'container_id':CID,'image_id':IMAGE,'shared_run_id':SHARED_RID,'model_parent':parents[0],'source_probe':{'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr},'model_requests':0,'SDK_init_calls':0,'remote_mutations':False}))
'''
def render(src,values):
 import re
 src=re.sub(r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b',lambda m:repr(values[m[0]]),src);ast.parse(src);return src
def plans():
 cfg=ROOT/'config/C32-packed-normal-runtime-v1/L04-after-control-runtime-v1.json';assert hashlib.sha256(cfg.read_bytes()).hexdigest()=='2136bdd5c6dc2d91c9d2c9ac0049f5acad72ee9dc7436c2b303df56f03c01bb9';sg=json.loads(cfg.read_text())
 v=ROOT/'results/vllm-formal-C32-v2-native-v2.json';assert json.loads(v.read_text())['pass'];vc=json.loads((ROOT/'config/vllm-correctness-v10/node-43.json').read_text())
 vr='/vllm-workspace/vllm';ar='/vllm-workspace/vllm-ascend';sr='/sgl-workspace/sglang/python/sglang';site='/usr/local/python3.12.13/lib/python3.12/site-packages'
 vp=[ar+'/vllm_ascend/'+p for p in ['ascend_forward_context.py','ascend_config.py','config.py','ops/fused_moe/moe_comm_method.py','ops/fused_moe/fused_moe.py','worker/model_runner_v1.py','distributed/parallel_state.py']]+[vr+'/vllm/'+p for p in ['config/parallel.py','v1/spec_decode/mtp.py']]
 sp=[sr+'/srt/'+p for p in ['server_args.py','arg_groups/pipeline.py','arg_groups/pd_disaggregation_hook.py','arg_groups/cuda_graph_hook.py','arg_groups/platform_hook.py','arg_groups/moe_hook.py','hardware_backend/npu/utils.py','distributed/bootstrap.py','distributed/parallel_state.py','mem_cache/kv_cache_configurator.py','model_executor/pool_configurator.py','model_executor/model_runner.py','speculative/eagle_worker_v2.py']]+[site+'/deep_ep/'+p for p in ['buffer.py','ep_strategy.py','strategies/low_latency_strategy.py']]
 vp=[ar+'/vllm_ascend/'+p for p in ['ops/fused_moe/token_dispatcher.py','ops/fused_moe/prepare_finalize.py','device/hardware_profile.py','envs.py']]
 return [('vllm',43,{'container_id':json.loads(v.read_text())['runtime_CIDs']['43'],'name':'glm53-vllm-shared-v4-20261017-43'},vc['image_id'],'20261017T1766-vllm-shared-pool-v4',[vr,ar],vp)]

def main():
 recovery=json.loads((ROOT/RECOVERY/'receipt.json').read_text());assert recovery['pass'] and recovery['formal_result_exists']
 raw=json.loads((ROOT/RECOVERY/'node-43.json').read_text());remote=json.loads(raw['stdout']);assert remote['owned_client_container_inspect']['rc']==1 and remote['owned_docker_client_process']['rc']==1,'Owned benchmark client must have exited'
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
