#!/usr/bin/env python3
"""Finite isolated dump-helper capability check; no model/service mutations."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

W=Path(__file__).resolve().parent
api=json.loads((W/'stage-dump-contract/installed-dump-contract.json').read_text())
assert api['version']=='2.10.0.post7'
source=api['files']['torch_npu/npu/graphs.py']
assert hashlib.sha256((W/'stage-dump-contract/installed-dump-source/torch_npu/npu/graphs.py').read_bytes()).hexdigest()==source['sha256']
run_id='dump-capability-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
remote_dir='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/'+run_id
TEST=r'''
import datetime,hashlib,importlib.metadata,json,os,pathlib,shutil,threading,time,traceback
root=pathlib.Path(OUTPUT);root.mkdir(exist_ok=False)
for name in ('cache','work','logs','torch-cache'):(root/name).mkdir()
os.environ.update({'ASCEND_RT_VISIBLE_DEVICES':'0','ASCEND_PROCESS_LOG_PATH':str(root/'logs'),
 'ASCEND_CACHE_PATH':str(root/'cache'),'ASCEND_WORK_PATH':str(root/'work'),'ASCEND_SLOG_PRINT_TO_STDOUT':'0',
 'ASCEND_GLOBAL_LOG_LEVEL':'3','PYTHONDONTWRITEBYTECODE':'1','TORCHINDUCTOR_CACHE_DIR':str(root/'torch-cache')})
d=importlib.metadata.distribution('torch_npu');assert d.version=='2.10.0.post7'
p=pathlib.Path(d.locate_file('torch_npu/npu/graphs.py'));assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED_SOURCE_SHA
timer=threading.Timer(50,lambda:os._exit(124));timer.daemon=True;timer.start()
state={'kind':'SYNTHETIC_OBSERVER_CAPABILITY_ONLY_NOT_MODEL_COUNTS','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'npu_device':0,'shape':[24],'dtype':'int32','replay_input_values':[3,8,1],'planned_replays':3,'observations':[],
 'target_source_or_service_changed':False,'model_inference_started':False}
try:
 import torch,torch_npu
 torch.npu.set_device(0)
 helper=getattr(torch_npu,'save_npugraph_tensor',None)
 state['entry']='torch_npu.save_npugraph_tensor' if helper is not None else 'torch.ops.npu.save_npugraph_tensor.default'
 if helper is None:helper=torch.ops.npu.save_npugraph_tensor.default
 src=torch.full((24,),-1,dtype=torch.int32,device='npu:0');out=torch.empty_like(src)
 out.copy_(src);helper(out,save_path=str(root/'warmup.pt'));torch.npu.synchronize()
 graph=torch.npu.NPUGraph()
 with torch.npu.graph(graph):
  out.copy_(src)
  helper(out,save_path=str(root/'replay.pt'))
 for ordinal,value in enumerate([3,8,1]):
  src.fill_(value);graph.replay();torch.npu.synchronize()
  deadline=time.monotonic()+3;observed=None;last_error=None
  while time.monotonic()<deadline:
   candidates=sorted(root.glob('replay*.pt'),key=lambda x:x.stat().st_mtime_ns,reverse=True)
   assert len(candidates)<=8
   for file in candidates:
    try:
     tensor=torch.load(file,map_location='cpu',weights_only=True)
     if isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==(24,) and tensor.dtype==torch.int32 and bool((tensor==value).all()):
      archive=root/('observed_replay_'+str(ordinal)+'.pt');shutil.copy2(file,archive)
      observed={'ordinal':ordinal,'expected_value':value,'observed_values':tensor.tolist(),'source_filename':file.name,
       'archive_filename':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()};break
    except Exception as error:last_error=repr(error)
   if observed is not None:break
   time.sleep(.05)
  assert observed is not None,{'ordinal':ordinal,'expected':value,'last_error':last_error}
  state['observations'].append(observed)
  assert sum(x.stat().st_size for x in root.glob('*.pt'))<=1024*1024
 state['status']='PASS_CAPABILITY_SMOKE_ONLY'
 state['raw_replay_filenames']=[x.name for x in root.glob('replay*.pt')]
except Exception:
 state['status']='FAIL_CAPABILITY_SMOKE';state['error']=traceback.format_exc()
finally:
 state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (root/'result.json').write_text(json.dumps(state,indent=2)+'\n');timer.cancel()
print(json.dumps(state,indent=2));raise SystemExit(0 if state['status']=='PASS_CAPABILITY_SMOKE_ONLY' else 1)
'''
HOST=r'''
import datetime,json,re,subprocess
base='wio-vllm-v41-repro169-20261028';stage='wio-vllm-v41-repro169-stage-20261028'
ids={base:'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092',stage:'42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602'}
image='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
def run(args,timeout=25):return subprocess.run(args,text=True,capture_output=True,timeout=timeout,check=False)
def inspect(name):
 r=run(['docker','inspect',name]);assert r.returncode==0,r.stderr;d=json.loads(r.stdout)[0]
 assert d['Id']==ids[name] and d['Image']==image
 return {'Id':d['Id'],'Image':d['Image'],'State':d['State']}
def npu():
 r=run(['docker','exec',stage,'npu-smi','info']);assert r.returncode==0,r.stderr
 assert all('No running processes found in NPU '+str(i) in r.stdout for i in range(8)),r.stdout
 return r.stdout
out={'kind':'SYNTHETIC_DUMP_CAPABILITY_SMOKE_ONLY','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'before':{base:inspect(base),stage:inspect(stage)},'output_directory':OUTPUT,'no_model_load_or_service_restart':True}
assert not out['before'][base]['State']['Running'] and out['before'][stage]['State']['Running']
out['before_npu_smi']=npu()
result=run(['docker','exec',stage,'timeout','--kill-after=5s','60s','python3','-c',TEST_SOURCE],timeout=80)
out['test']={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
out['after']={base:inspect(base),stage:inspect(stage)}
out['after_npu_smi']=npu()
assert not out['after'][base]['State']['Running'] and out['after'][stage]['State']['Running']
out['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out,indent=2))
raise SystemExit(0 if result.returncode==0 else 1)
'''
test='OUTPUT='+repr(remote_dir)+'\nEXPECTED_SOURCE_SHA='+repr(source['sha256'])+'\n'+TEST
host='OUTPUT='+repr(remote_dir)+'\nTEST_SOURCE='+repr(test)+'\n'+HOST
payload=base64.b64encode(host.encode()).decode()
command=shlex.join(['python3','-c',"import base64;exec(base64.b64decode('"+payload+"'))"])
with (W/'graph-dump-capability-smoke.json').open('x') as out,(W/'graph-dump-capability-smoke.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=130,check=False)
print(json.dumps({'returncode':result.returncode,'run_id':run_id,'remote_output_directory':remote_dir,
    'scope':'one device,24 int32 elements,one warmup and3 replays; not model workload,performance or expert-count evidence'}))
sys.exit(result.returncode)
