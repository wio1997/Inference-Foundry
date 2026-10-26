"""Exact-once fixed-image no-NPU loader metadata prerequisite; models unchanged."""
import ast,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12C-same-inode-native-loader-CPU-v1'
HOST=r'''import pathlib,subprocess,json,hashlib,shutil,urllib.request
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261019T-L12B-rollback-v1' and d['State']['Running']
state=json.loads((pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261019T-L12B-rollback-v1')/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and 'model_exit_code' not in state
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for n in [34,35,36,41]:
 port=30500 if n in [34,35] else 30600
 with opener.open('http://172.16.12.'+str(n)+':'+str(port)+'/health',timeout=10) as f:assert f.status==200
 with opener.open('http://172.16.12.'+str(n)+':'+str(port)+'/metrics',timeout=10) as f:raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2
 values=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert values and not any(values)
def owned():
 r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0;pids=[]
 for line in r.stdout.split('Process id',1)[-1].splitlines():
  c=[v.strip() for v in line.split('|')]
  if len(c)>=5 and len(c[1].split())==2 and all(v.isdigit() for v in c[1].split()) and c[2].isdigit():
   pid=int(c[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();pids.append(pid)
 return sorted(pids)
before=owned();assert len(before)==18
root=pathlib.Path(REMOTE);assert not root.exists();root.mkdir(parents=True);loader=root/'loader';loader.mkdir()
source=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L12-native-HiCache-extension-build-v3/extension/libmf_hybm_accoffload.so');assert hashlib.sha256(source.read_bytes()).hexdigest()=='3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46'
shutil.copyfile(source,loader/'libmf_hybm_accoffload.so');(loader/'libmf_acc_offload.so').symlink_to('/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_acc_offload.so')
for p in ['mf-logs','mc-logs','ascend']:(root/p).mkdir()
container_env=dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
args=['docker','run','-i','--rm','--pull=never','--network=none','--read-only','--cpus=2','--memory=4g','--pids-limit=256','--tmpfs','/tmp:rw,size=512m','--label','project_id=glm53','--label','run_id=20261019T-L12C-same-inode-native-loader-CPU-v1','--name','glm53-L12C-native-loader-CPU-v1-20261019-34','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v',str(loader)+':/opt/glm53/native-loader:ro','-v',str(root/'mf-logs')+':/var/log/memfabric_hybrid','-v',str(root/'mc-logs')+':/var/log/memcache_hybrid','-v',str(root/'ascend')+':/root/ascend','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/lib:'+container_env.get('LD_LIBRARY_PATH',''),'--entrypoint','/usr/bin/timeout',IMAGE,'--signal=TERM','--kill-after=5s','60s','python3','-']
r=subprocess.run(args,input=PROBE,capture_output=True,text=True,timeout=85);after=owned();out={'project_id':'glm53','node':34,'CPU_rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'original_NPU_context_PIDs_before':before,'original_NPU_context_PIDs_after':after,'original_NPU_contexts_unchanged':before==after,'existing_models_changed':False,'pass':r.returncode==0 and before==after}
assert len(r.stdout.encode())+len(r.stderr.encode())<=2*1024**2
print(json.dumps(out))
'''
def plan():
 p=ROOT/'scripts/glm53_C32_L12B_P_transition_library_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='0f59ff934fa11980d17b590948250b22baffb088cbbe4b2daab2493e6545d146';s=importlib.util.spec_from_file_location('lib',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 cap=json.loads((ROOT/'results/sglang-C32-L12B-restored-L10-P-pool-capacity-v1.json').read_text());p=ROOT/'scripts/qualify_C32_L12C_same_inode_native_loader_CPU_v1.py';probe=p.read_text();ast.parse(probe);item=cap['providers']['34'];code=m.replace(HOST,{'CID':item['container_id'],'NAME':item['name'],'IMAGE':m.CURRENT['model_image'],'REMOTE':'/data/wio/inference-foundry/glm53/runs/'+RID,'PROBE':probe});ast.parse(code);return m,code

def main():
 evidence=ROOT/'results/sglang-C32-L12B-restored-L10-native-readiness-v1.json';assert hashlib.sha256(evidence.read_bytes()).hexdigest()=='3eb2d87f52356c8bfc6f36d2f16724f3130ec01f9b64eca11292006fd4a77995' and json.loads(evidence.read_text())['pass']
 m,code=plan();review=json.loads((ROOT/'results/C32-L12C-same-inode-loader-CPU-actual-review-v1.json').read_text());assert review['host_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'models_changed':False,'max_SSH':1}
 try:
  r=m.stage().invoke(34,code,150);(run/'node-34.json').write_text(json.dumps(r,indent=2)+'\n');receipt['node_rc']=r['rc'];assert r['rc']==0;d=json.loads(r['stdout']);receipt['CPU_rc']=d['CPU_rc'];receipt['pass']=d['pass'];receipt['original_NPU_contexts_unchanged']=d['original_NPU_contexts_unchanged']
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
