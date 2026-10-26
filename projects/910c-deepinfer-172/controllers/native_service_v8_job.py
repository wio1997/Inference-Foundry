from pathlib import Path
import subprocess,json,time,re
p=Path('/data/tiankuan/wio/910c-deepinfer-172');status=p/'runs/serving-v8-status.json';control=p/'serving/v8'
assert not status.exists(),'already dispatched'
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert 'Process id' in smi and not re.search(r'^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|',smi,re.M),'occupied; refuse start'
port=subprocess.check_output(['ss','-ltnp','sport = :8172'],text=True)
assert len(port.strip().splitlines())==1,'8172 already occupied; refuse start'
control.mkdir(parents=True,exist_ok=False)
env='PYTHONPATH='+str(p/'src/DeepInfer')+':/usr/local/Ascend/cann-9.2.0/python/site-packages:/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe'
worker_cmd=['docker','exec','-e','DEEPINFER_910C_MC2=1','-e',env,'-w',str(p/'src/DeepInfer'),'wio-deepinfer-s172-npu','python','-m','torch.distributed.run','--standalone','--nproc-per-node=16','examples/ascend910c_w8a8_worker.py','--model-dir','/models/DeepSeek-V4.1-Flash-1017-w8a8','--control-dir',str(control)]
api_cmd=['docker','exec','-e',env,'-w',str(p/'src/DeepInfer'),'wio-deepinfer-s172','python','examples/ascend910c_w8a8_api.py','--model-dir','/models/DeepSeek-V4.1-Flash-1017-w8a8','--control-dir',str(control)]
s={'state':'LOADING','started':time.time(),'worker_command':worker_cmd,'api_command':api_cmd,'source_manifest':'evidence/serving-v8-source/SHA256.json'}
def save():status.write_text(json.dumps(s,indent=2))
with (p/'logs/serving-v8-worker.log').open('w') as worker_log,(p/'logs/serving-v8-api.log').open('w') as api_log:
 worker=subprocess.Popen(worker_cmd,stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True);s['worker_host_pid']=worker.pid;save()
 while len(list(control.glob('ready-rank*.json')))<16 and worker.poll() is None and time.time()-s['started']<600:time.sleep(2)
 if worker.poll() is not None or len(list(control.glob('ready-rank*.json')))<16:
  s.update(state='FAIL',worker_exit=worker.poll(),reason='native load failed or timed out');save();raise SystemExit(1)
 api=subprocess.Popen(api_cmd,stdout=api_log,stderr=subprocess.STDOUT,start_new_session=True);s.update(state='HTTP_STARTING',api_host_pid=api.pid);save()
 import urllib.request
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for attempt in range(60):
  try:
   with op.open('http://127.0.0.1:8172/health',timeout=2) as r:health=json.load(r)
   assert health['backend']=='DeepInfer-native';s.update(state='READY',health=health);save();break
  except Exception:
   if worker.poll() is not None or api.poll() is not None:break
   time.sleep(1)
 else:s.update(state='FAIL',reason='HTTP readiness timeout');save();raise SystemExit(1)
 while worker.poll() is None and api.poll() is None:time.sleep(3)
 s.update(state='FAIL',worker_exit=worker.poll(),api_exit=api.poll(),reason='resident process exited');save()
