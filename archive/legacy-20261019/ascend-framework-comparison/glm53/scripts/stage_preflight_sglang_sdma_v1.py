#!/usr/bin/env python3
"""Stage exact own source overlays; native import preflight only, no engines."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0960-sglang-sdma-preflight-v1';REMOTE='/data/wio/inference-foundry/glm53/overlays/sglang-sdma-v1-20261017';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
ROOT=b.ROOT/'config/overlays/sglang-sdma-v1';assert hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()=='0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528'
MANIFEST=json.loads((ROOT/'manifest.json').read_text());FILES={}
for entry in MANIFEST['files']+MANIFEST['helpers']:
 raw=(ROOT/entry['local']).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry.get('patched_sha256',entry.get('sha256'));FILES[entry['local']]=raw.decode()
FILES['manifest.json']=(ROOT/'manifest.json').read_text()
FILES['config/meta.conf']='ock.mmc.meta_service_url = tcp://172.16.12.34:29780\nock.mmc.meta_service.config_store_url = tcp://172.16.12.34:29781\nock.mmc.meta_service.metrics_url = http://172.16.12.34:29782\nock.mmc.log_level = info\n'
FILES['config/local.conf']='ock.mmc.meta_service_url = tcp://172.16.12.34:29780\nock.mmc.local_service.config_store_url = tcp://172.16.12.34:29781\nock.mmc.log_level = info\nock.mmc.local_service.world_size = 128\nock.mmc.local_service.protocol = device_sdma\nock.mmc.local_service.dram.size = 24GB\n'
CONFIG_SHA={'34':'185bc2817a19ed008b0213d22e7c4d1da5ed3d6f6ce9ec32d2dc1494517f83eb','35':'b0146297a569cf8c0ae3c47ac8148baf801edf567a21f44d0d32df8a8ff5781c','36':'4272bac65b89232fb36e08f3fca3aa2a6b806b18b078df1784a545b7791eb1be','41':'3ad954253ecd69221d0b3822b4b53698f9f363a532639167086cc6422ee67553'}
for n,sha in CONFIG_SHA.items():
 raw=(b.ROOT/f'config/sglang-correctness-v2/node-{n}.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;FILES[f'config/node-{n}.json']=raw.decode()
def invoke(node,code,timeout=150):
 pod,host=b.NODES[node];mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def stage(node):
 check='import pathlib,hashlib,json;items='+repr(MANIFEST['files'])+';print(json.dumps({e["container_path"]:hashlib.sha256(pathlib.Path(e["container_path"]).read_bytes()).hexdigest() for e in items}))'
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=1','--memory=1g','--pids-limit=128','--label','project_id=glm53','--name',f'glm53-sglang-source-check-20261017-{node}','--entrypoint','python3',IMAGE,'-c',check]
 code=r'''import pathlib,subprocess,json,hashlib,sys
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==IMAGE
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
actual=json.loads(r.stdout);assert actual=={x['container_path']:x['original_sha256'] for x in MANIFEST['files']}
root.mkdir(parents=True)
for name,text in FILES.items():
 path=root/name;assert '..' not in pathlib.PurePosixPath(name).parts;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256(text.encode()).hexdigest()
print(json.dumps({'node':NODE,'image':IMAGE,'source_sha_original':actual,'staged_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in FILES.items()}}))
'''
 for k,v in [('REMOTE',REMOTE),('IMAGE',IMAGE),('ARGS',args),('MANIFEST',MANIFEST),('FILES',FILES),('NODE',node)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return invoke(node,code)
def preflight():
 code=r'''import pathlib,hashlib,json,inspect,os
root=pathlib.Path('/opt/glm53/source');manifest=json.loads((root/'manifest.json').read_text())
for e in manifest['files']:assert hashlib.sha256(pathlib.Path(e['container_path']).read_bytes()).hexdigest()==e['patched_sha256']
for e in manifest['helpers']:assert hashlib.sha256((root/e['local']).read_bytes()).hexdigest()==e['sha256']
import glm53_sglang_sdma_audit as guard
guard.prepare_sdk()
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.ascend.conn import AscendKVManager
assert AscendTransferEngine._get_transfer_protocol()=='sdma' and AscendTransferEngine._resolve_trans_op_type('sdma').value==1
assert '_glm53.transfer' in inspect.getsource(AscendTransferEngine.batch_transfer_sync)
assert '_glm53.registered' in inspect.getsource(AscendTransferEngine.batch_register)
assert '_glm53.reject_tcp' in inspect.getsource(AscendKVManager.send_aux_tcp)
assert '_glm53.reject_tcp' in inspect.getsource(AscendKVManager._handle_aux_data)
from sglang.srt.environ import envs
assert envs.SGLANG_MOONCAKE_SEND_AUX_TCP.get() is False
print('GLM53_SGLANG_SDMA_PREFLIGHT='+json.dumps({'imports_pass':True,'sdk_guard':True,'native_engine_created':False,'model_requests':0,'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}),flush=True)
'''
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=4','--memory=8g','--shm-size=1g','--pids-limit=1024','--label','project_id=glm53','--name','glm53-sglang-sdma-preflight-20261017-34','-v',REMOTE+':/opt/glm53/source:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
 for e in MANIFEST['files']:args+=['-v',REMOTE+'/'+e['local']+':'+e['container_path']+':ro']
 for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
 for key,value in {'PYTHONPATH':'/opt/glm53/source/python','ASCEND_MF_TRANSFER_PROTOCOL':'sdma','SGLANG_MOONCAKE_SEND_AUX_TCP':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}.items():args+=['-e',key+'='+value]
 boot='set -eu; mkdir -p /run/glm53/audit; exec python3 -c "$1"'
 args+=['--entrypoint','/bin/bash',IMAGE,'-c',boot,'glm53-preflight',code]
 host=r'''import subprocess,json,sys
before=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert before.returncode==0 and before.stdout.count('No running processes')==8
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=180);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='')
after=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);print('GLM53_POSTCHECK_JSON='+json.dumps({'rc':after.returncode,'npu_after':after.stdout}),flush=True)
assert after.returncode==0 and after.stdout.count('No running processes')==8
sys.exit(r.returncode)
'''
 return invoke(34,host.replace('ARGS',repr(args)),270)
def main():
 prior=json.loads((b.ROOT/'runs/20261017T0910-sglang-config-preflight-v4/receipt.json').read_text());assert prior['rc']==0
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'stage_rc':{},'preflight_rc':None,'model_requests':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(stage,n):n for n in (34,35,36,41)}
   for future in concurrent.futures.as_completed(futures):
    node=futures[future]
    try:r=future.result()
    except Exception as error:r={'error':repr(error)}
    (RUN/f'stage-{node}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['stage_rc'][node]=r.get('rc',r.get('error'))
  assert all(x==0 for x in receipt['stage_rc'].values())
  r=preflight();(RUN/'preflight-34.json').write_text(json.dumps(r,indent=2)+'\n');receipt['preflight_rc']=r['rc']
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
