"""One bounded private process in owned current P container; no model restart or SDK change."""
import ast,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12C-combined-MC-native-fixture-v1'
HOST=r'''import pathlib,subprocess,json,hashlib,shutil,time,urllib.request
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261019T-L12B-rollback-v1' and d['State']['Running'] and not d['HostConfig']['Privileged']
shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261019T-L12B-rollback-v1');sup=json.loads((shared/'supervisor-status.json').read_text());assert sup['phase']=='MODEL_RUNNING' and 'model_exit_code' not in sup
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def health():
 result={}
 for n in [34,35,36,41]:
  url='http://172.16.12.'+str(n)+':'+str(30500 if n in [34,35] else 30600)
  with opener.open(url+'/health',timeout=10) as f:assert f.status==200
  with opener.open(url+'/metrics',timeout=10) as f:raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2
  gauges=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges);result[str(n)]={'healthy_idle':True,'metrics_sha256':hashlib.sha256(raw).hexdigest()}
 return result
def owned():
 r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0;pids=[]
 for line in r.stdout.split('Process id',1)[-1].splitlines():
  c=[x.strip() for x in line.split('|')]
  if len(c)>=5 and len(c[1].split())==2 and all(x.isdigit() for x in c[1].split()) and c[2].isdigit():
   pid=int(c[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();pids.append(pid)
 return sorted(pids)
before_health=health();before=owned();assert len(before)==18
root=shared/RID;assert not root.exists();root.mkdir();loader=root/'loader';loader.mkdir()
source=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L12-native-HiCache-extension-build-v3/extension/libmf_hybm_accoffload.so');raw=source.read_bytes();assert len(raw)==718752 and hashlib.sha256(raw).hexdigest()=='3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46'
(loader/'libmf_hybm_accoffload.so').write_bytes(raw);(loader/'libmf_acc_offload.so').symlink_to('/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_acc_offload.so')
case='/run/glm53/'+RID;container_env=dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
env={'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','GLM53_NODE':'34','SGLANG_HICACHE_HOST_MEM_BACKEND':'memfabric','GLM53_C32_NATIVE_HICACHE_MTP':'1','GLM53_NATIVE_LOADER_DIR':case+'/loader','GLM53_NATIVE_CASE_ROOT':case,'LD_LIBRARY_PATH':'/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/lib:'+container_env.get('LD_LIBRARY_PATH','')}
args=['docker','exec','-i',CID,'env']+[k+'='+v for k,v in env.items()]+['/usr/bin/timeout','--signal=TERM','--kill-after=15s','240s','python3','-']
p=subprocess.run(args,input=PROBE,capture_output=True,text=True,timeout=280);assert len(p.stdout.encode())+len(p.stderr.encode())<=4*1024**2
out={'project_id':'glm53','node':34,'container_id':CID,'model_requests':0,'models_changed':False,'fixture_rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'owned_context_PIDs_before':before,'all4_health_before':before_health,'private_process_timeout_seconds':240,'private_host_reserve_bytes':1024**3,'fixture_device_tensor_bytes':56819712,'maximum_own_temporary_MC_bytes':2*71024640+1024**2}
(root/'fixture-stdout.txt').write_text(p.stdout);(root/'fixture-stderr.txt').write_text(p.stderr)
deadline=time.monotonic()+30
while True:
 after=owned()
 if after==before or time.monotonic()>=deadline:break
 time.sleep(2)
out['owned_context_PIDs_after']=after;out['original18contexts_unchanged']=after==before;out['all4_health_after']=health()
for name in ['step.json','keys.json']:
 f=root/name
 if f.is_file():out[name]=json.loads(f.read_text())
rows=[x.split('=',1)[1] for x in p.stdout.splitlines() if x.startswith('GLM53_MC_OFFLOAD_COMBINED_MRE_RESULT=')]
if len(rows)==1:out['fixture_result']=json.loads(rows[0])
out['pass']=p.returncode==0 and after==before and len(rows)==1 and out['fixture_result']['pass']
print(json.dumps(out))
'''
def plan():
 p=ROOT/'scripts/glm53_C32_L12B_P_transition_library_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='0f59ff934fa11980d17b590948250b22baffb088cbbe4b2daab2493e6545d146';s=importlib.util.spec_from_file_location('lib',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 compat=ROOT/'scripts/glm53_C32_hicache_native_mtp_v1.py';assert hashlib.sha256(compat.read_bytes()).hexdigest()=='50d7cc8040b9956092dd40fe204bc5d6d3633bf04f1ba6937a8273fc72ac0b2f'
 bootstrap=ROOT/'scripts/glm53_C32_L12C_same_inode_sdk_bootstrap_v1.py';assert hashlib.sha256(bootstrap.read_bytes()).hexdigest()=='403bca635c37e7010edd65ec2af86ba8066ce32838b9b961c2381e0108639d8b'
 fixture=ROOT/'scripts/glm53_C32_L12C_combined_MC_native_fixture_v1.py';probe='LOADER_BOOTSTRAP_SOURCE='+repr(bootstrap.read_text())+'\nCOMPATIBILITY_SOURCE='+repr(compat.read_text())+'\n'+fixture.read_text();ast.parse(probe)
 cap=json.loads((ROOT/'results/sglang-C32-L12B-restored-L10-P-pool-capacity-v1.json').read_text());item=cap['providers']['34'];code=m.replace(HOST,{'CID':item['container_id'],'NAME':item['name'],'IMAGE':m.CURRENT['model_image'],'RID':RID,'PROBE':probe});ast.parse(code);return m,code

def main():
 prerequisite=json.loads((ROOT/'results/C32-L12C-same-inode-loader-CPU-qualified-v1.json').read_text());assert prerequisite['pass'] and not prerequisite['native_initialization_or_combined_MC_correctness_proven']
 assert json.loads((ROOT/'results/C32-L12C-MC-Offload-existing-mapping-source-review-v1.json').read_text())['model_reload_allowed'] is False
 m,code=plan();review=json.loads((ROOT/'results/C32-L12C-combined-MC-native-fixture-actual-review-v1.json').read_text());assert review['host_code_sha256']==hashlib.sha256(code.encode()).hexdigest()
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','model_requests':0,'max_SSH':1,'models_changed':False}
 try:
  r=m.stage().invoke(34,code,390);(run/'node-34.json').write_text(json.dumps(r,indent=2)+'\n');receipt['node_rc']=r['rc'];assert r['rc']==0;d=json.loads(r['stdout']);receipt['fixture_rc']=d['fixture_rc'];receipt['pass']=d['pass'];receipt['original18contexts_unchanged']=d['original18contexts_unchanged']
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
