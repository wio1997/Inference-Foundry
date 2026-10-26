"""Exact-once ZCode launcher: one private bounded native IO fixture on owned34."""
import ast,hashlib,importlib.util,json,os,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L12-C32-native-HiCache-NPU-fixture-v4'
HOST=r'''
import socket,pathlib,subprocess,json,time,re,urllib.request,hashlib
assert socket.gethostname()==HOSTNAME
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0];assert d['Id']==CID and d['Image']==IMAGE and d['Name']=='/'+NAME and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED
state=json.loads((pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and 'model_exit_code' not in state
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));idle={}
for n in [34,35,36,41]:
 url='http://172.16.12.'+str(n)+(':'+str(30500 if n in [34,35] else 30600))
 with opener.open(url+'/health',timeout=10) as f:assert f.status==200
 with opener.open(url+'/metrics',timeout=10) as f:raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2
 values=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert values and all(x==0 for x in values);idle[str(n)]=values
def owned():
 p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert p.returncode==0;rows=[]
 for line in p.stdout.split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
   path=pathlib.Path('/proc')/cols[2];assert CID in (path/'cgroup').read_text();rows.append(int(cols[2]))
 return sorted(rows)
before=owned();assert len(before)==18
extension_source=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L12-native-HiCache-extension-build-v3/extension/libmf_hybm_accoffload.so')
extension_raw=extension_source.read_bytes();assert hashlib.sha256(extension_raw).hexdigest()=='3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46' and len(extension_raw)==718752
fixture_root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L12-C32-native-HiCache-NPU-fixture-v4');fixture_root.mkdir(exist_ok=False)
for name in ['mf-logs','mc-logs','ascend','cache']:(fixture_root/name).mkdir()
out={'project_id':'glm53','node':34,'container_id':CID,'model_requests':0,'all4_idle_before':idle,'owned_NPU_context_PIDs_before':before,'fixture_timeout_enforced_inside_container':True}
container_env=dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
args=['docker','run','-i','--rm','--pull=never','--read-only','--network=none','--cpus=4','--memory=8g','--pids-limit=1024','--shm-size=1g','--ulimit','memlock=-1','--tmpfs','/tmp:rw,size=2g','--label','project_id=glm53','--label','run_id=20261018T-L12-C32-native-HiCache-NPU-fixture-v4','--name','glm53-C32-native-HiCache-fixture-v4-20261018-34','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v',str(fixture_root)+':/run/glm53','-v',str(fixture_root/'mf-logs')+':/var/log/memfabric_hybrid','-v',str(fixture_root/'mc-logs')+':/var/log/memcache_hybrid','-v',str(fixture_root/'ascend')+':/root/ascend','-v',str(fixture_root/'cache')+':/root/.cache','-v',str(extension_source)+':/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so:ro']
for path in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
 assert pathlib.Path(path).exists();args+=['--device',path]
env={'PYTHONDONTWRITEBYTECODE':'1','ASCEND_VISIBLE_DEVICES':'0','ASCEND_RT_VISIBLE_DEVICES':'0','ACL_OP_INIT_MODE':'1','ASCEND_ENABLE_USE_FABRIC_MEM':'1','GLM53_C32_NATIVE_HICACHE_MTP':'1','SGLANG_HICACHE_HOST_MEM_BACKEND':'memfabric','GLM53_NATIVE_EXTENSION_SHA256':'3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46','LD_LIBRARY_PATH':'/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib:'+container_env.get('LD_LIBRARY_PATH',''),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
for key,value in env.items():args+=['-e',key+'='+value]
args+=['--entrypoint','/usr/bin/timeout',IMAGE,'--signal=TERM','--kill-after=15s','240s','python3','-']
out['fixture_private_container_argv']=args
out['optional_extension_readonly_container_mount']=True
p=subprocess.run(args,input=PROBE,capture_output=True,text=True,timeout=280);out['fixture_rc']=p.returncode;out['stdout']=p.stdout;out['stderr']=p.stderr
assert len(p.stdout.encode())+len(p.stderr.encode())<=2*1024**2
deadline=time.monotonic()+30
while True:
 after=owned()
 if after==before or time.monotonic()>=deadline:break
 time.sleep(2)
out['owned_NPU_context_PIDs_after']=after;out['original_NPU_contexts_unchanged']=after==before
out['pass']=p.returncode==0 and after==before
if p.returncode==0:
 rows=[x.split('=',1)[1] for x in p.stdout.splitlines() if x.startswith('GLM53_NATIVE_HICACHE_FIXTURE_RESULT=')];assert len(rows)==1
 out['fixture_result']=json.loads(rows[0]);assert out['fixture_result']['pass']
print(json.dumps(out))
'''
def plan():
    compat=ROOT/'scripts/glm53_C32_hicache_native_mtp_v1.py';fixture=ROOT/'scripts/glm53_C32_native_hicache_NPU_fixture_v3.py'
    assert hashlib.sha256(compat.read_bytes()).hexdigest()=='50d7cc8040b9956092dd40fe204bc5d6d3633bf04f1ba6937a8273fc72ac0b2f'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='4c0a2658c1038c161a69ba31a598009f39f80ff6ddb812f9d593e3a60c7206d2'
    probe='COMPATIBILITY_SOURCE='+repr(compat.read_text())+'\n'+fixture.read_text();ast.parse(probe)
    settings=json.loads((ROOT/'config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v2.json').read_text());item=settings['runtime_containers']['34']
    values={'HOSTNAME':'B5-15U-Atlas900RCKA3-34','CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'SHARED':settings['shared_run_id'],'PROBE':probe}
    code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST);ast.parse(code)
    return code,values
def main():
    proof=json.loads((ROOT/'results/C32-L12-native-HiCache-MTP-CPU-review-v1.json').read_text());assert proof['pass']
    binary=json.loads((ROOT/'results/C32-L12-private-native-extension-qualified-build-v1.json').read_text());assert binary['build_artifact_valid_for_native_fixture'] and binary['library_sha256']=='3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46'
    path=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
    spec=importlib.util.spec_from_file_location('inventory',path);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()))
    receipt={'project_id':'glm53','model_requests':0,'fixture_attempted':False,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        code,values=plan();(run/'reviewed-host-code.py').write_text(code)
        pod=b.NODES[34][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.34'
        receipt['fixture_attempted']=True
        cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
        result=b.invoke([cmd],code,timeout=360);(run/'node-34.json').write_text(json.dumps(result,indent=2)+'\n');receipt['node_rc']=result['rc']
        if result['rc']==0:
            obj=json.loads(result['stdout']);receipt['fixture_rc']=obj['fixture_rc'];receipt['pass']=obj['pass'];receipt['original_NPU_contexts_unchanged']=obj['original_NPU_contexts_unchanged']
    finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
