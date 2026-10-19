#!/usr/bin/env python3
"""Old/new env enumeration only; no model or MemCache initialization."""
import ast,hashlib,importlib.util,json,re
from pathlib import Path
P=Path(__file__).resolve();dep=P.parent/'probe_sglang_standalone_allocation_v1.py'
assert hashlib.sha256(dep.read_bytes()).hexdigest()=='8338fb6c8666d9a2ccdcfe0183a3dd14eb095fa8dc79e3ecbacbb7740405ea77'
s=importlib.util.spec_from_file_location('base_probe',dep);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
RID='20261017T1680-sglang-npu-visibility-probe-v1';RUN=base.b.ROOT/'runs'/RID
IMAGE=base.IMAGE;PROVIDER='264d475e8173858cbcbee2769503ffd984124ff349c4e47d8f53afcb3bd0a59a';FAILED='0c03e9cb26976e0c3f082805670da920ce6ac730b5370220319572818c859345'
CODE='''import os,json,pathlib,hashlib
out={'selected_env':{k:os.environ.get(k) for k in ['ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','LD_LIBRARY_PATH']},'davinci_devices':sorted(str(p) for p in pathlib.Path('/dev').glob('davinci*')),'model_requests':0,'SDK_init_calls':0}
import torch,torch_npu
out['torch_version']=torch.__version__;out['torch_npu_version']=torch_npu.__version__
try:out['torch_device_count']=torch.npu.device_count();out['torch_is_available']=torch.npu.is_available()
except Exception as e:out['torch_count_error']=repr(e)
try:
 import acl
 out['acl_device_count']=acl.rt.get_device_count();out['acl_soc_name']=acl.get_soc_name()
except Exception as e:out['acl_count_error']=repr(e)
p=pathlib.Path(torch_npu.npu.__file__);raw=p.read_bytes();out['torch_npu_source']={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'lines340_425':raw.decode().splitlines()[340:425]}
print('GLM53_NPU_VISIBILITY='+json.dumps(out))
'''
HOST=r'''import subprocess,json,pathlib,re
def cmd(a,t=30):
 r=subprocess.run(a,capture_output=True,text=True,timeout=t);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
root=pathlib.Path(REMOTE);assert not root.exists();root.mkdir(parents=True)
out={'model_requests':0,'SDK_init_calls':0,'containers':{},'cases':{}}
for cid,expected_run in [(PROVIDER,'20261017T1645-sglang-four-node-pool-start-v1'),(FAILED,'20261017T1655-sglang-correctness-v7-v1'),('9844ec4193f7bc662ac43a62e2606a72f8299992283f27d2005a8ce9c643f805','20261017T1562-sglang-correctness-v6-v3')]:
 r=cmd(['docker','inspect',cid]);assert r['rc']==0;d=json.loads(r['stdout'])[0];assert d['Id']==cid and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==expected_run
 assert d['State']['Running'] is (cid==PROVIDER)
 out['containers'][cid]={'name':d['Name'],'state':d['State'],'Runtime':d['HostConfig'].get('Runtime'),'Privileged':d['HostConfig'].get('Privileged'),'Devices':d['HostConfig'].get('Devices'),'DeviceCgroupRules':d['HostConfig'].get('DeviceCgroupRules'),'selected_env':{k:v for e in d['Config']['Env'] for k,_,v in [e.partition('=')] if k in ['ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE']}}
provider_pid=out['containers'][PROVIDER]['state']['Pid']
def npu_check():
 r=cmd(['npu-smi','info']);rows=re.findall(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|',r['stdout'].split('Process id',1)[-1],re.M)
 assert r['rc']==0 and {int(x[2]) for x in rows}<={provider_pid} and len({x[0] for x in rows})+r['stdout'].count('No running processes')==8
 return r
out['npu_before']=npu_check()
for label,args in ARGS:
 out['cases'][label]=cmd(args,100);(root/(label+'.json')).write_text(json.dumps(out['cases'][label],indent=2))
out['npu_after']=npu_check();(root/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 commands=[]
 for label,env in [('baseline',{}),('documented',{'ACL_OP_INIT_MODE':'1','ASCEND_ENABLE_USE_FABRIC_MEM':'1'}),('visible_all',{'ACL_OP_INIT_MODE':'1','ASCEND_ENABLE_USE_FABRIC_MEM':'1','ASCEND_RT_VISIBLE_DEVICES':','.join(map(str,range(16)))})]:
  args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=8g','--pids-limit=512','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-visibility-v1-20261017-34-'+label,'-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1']
  for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
  for k,v in env.items():args+=['-e',k+'='+v]
  args+=['--entrypoint','python3',IMAGE,'-c',CODE];commands.append((label,args))
 code=base.replace(HOST,{'REMOTE':'/data/wio/inference-foundry/glm53/runs/'+RID,'PROVIDER':PROVIDER,'FAILED':FAILED,'IMAGE':IMAGE,'ARGS':commands});ast.parse(code)
 result=base.invoke(34,code,timeout=390);(RUN/'node-34.json').write_text(json.dumps(result,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':result['rc'],'model_requests':0,'SDK_init_calls':0},indent=2)+'\n')
if __name__=='__main__':main()
