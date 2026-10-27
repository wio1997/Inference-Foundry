from pathlib import Path
import subprocess,json,time,re,os
p=Path('/data/tiankuan/wio/910c-deepinfer-172');status=p/'runs/serving-dspark-v8-status.json';control=p/'serving/dspark-v8'
assert not status.exists(),'already dispatched'
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert 'Process id' in smi and not re.search(r'^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|',smi,re.M),'occupied; refuse start'
port=subprocess.check_output(['ss','-ltnp','sport = :8172'],text=True)
assert len(port.strip().splitlines())==1,'8172 already occupied; refuse start'
control.mkdir(parents=True,exist_ok=False)
env='PYTHONPATH='+str(p/'src/DeepInfer-dspark-v8')+':/usr/local/Ascend/cann-9.2.0/python/site-packages:/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe'
worker_cmd=['docker','exec','-e','DEEPINFER_910C_FUSED_MC2_DECODE=1','-e','DEEPINFER_910C_FUSED_O_PROJ_DECODE=1','-e','DEEPINFER_910C_STARTUP_GRAPHS=1','-e','DEEPINFER_910C_FUSED_ROPE=1','-e','DEEPINFER_910C_DSPARK=1','-e','DEEPINFER_USE_NEXTN_DECODE=5','-e','DEEPINFER_ENABLE_MTP_VARLEN=0','-e','DEEPINFER_PROMPT_LOOKUP_TOKENS=0','-e','DEEPINFER_910C_MC2=1','-e','DEEPINFER_910C_FUSED_MC2_PREFILL=1','-e','DEEPINFER_910C_SEQUENCE_PARALLEL=1','-e','DEEPINFER_910C_CONTEXT_PARALLEL=1','-e','DEEPINFER_910C_KV_TOKEN_PARALLEL=1','-e','DEEPINFER_910C_TAIL_FFN=1','-e','DEEPINFER_910C_FUSED_O_PROJ=1','-e',env,'-w',str(p/'src/DeepInfer-dspark-v8'),'wio-deepinfer-s172-npu','python','-m','torch.distributed.run','--standalone','--nproc-per-node=16','examples/ascend910c_w8a8_worker.py','--model-dir','/models/DeepSeek-V4.1-Flash-1017-w8a8','--control-dir',str(control),'--chunk-size','4096','--mc2-buffer-mib','1024']
api_cmd=['docker','exec','-e',env,'-w',str(p/'src/DeepInfer-dspark-v8'),'wio-deepinfer-s172','python','examples/ascend910c_w8a8_api.py','--model-dir','/models/DeepSeek-V4.1-Flash-1017-w8a8','--control-dir',str(control),'--tokenizer-cache-tokens','1048576']
s={'state':'LOADING','started':time.time(),'worker_command':worker_cmd,'api_command':api_cmd,'source_manifest':'evidence/dspark-v8-source/SHA256.json'}
def save():status.write_text(json.dumps(s,indent=2))
with (p/'logs/serving-dspark-v8-worker.log').open('w') as worker_log,(p/'logs/serving-dspark-v8-api.log').open('w') as api_log:
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

 (control/'jobs').mkdir(exist_ok=True)
 version='dspark-v8';parent=version;dest=p/'diagnostics'/version;dest.mkdir(exist_ok=True)
 smi=subprocess.check_output(['npu-smi','info'],text=True);pids=[int(x) for x in re.findall(r'^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|',smi,re.M)];assert len(pids)==16
 workers=[]
 for pid in pids:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'ascend910c_w8a8_worker.py' in cmd and '/serving/dspark-v8' in cmd,(pid,cmd)
  env=dict(x.split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in x);rank=int(env['LOCAL_RANK']);assert 0<=rank<16
  allowed=sorted(os.sched_getaffinity(pid));assert allowed==list(range(640)),('unexpected existing affinity',pid,len(allowed))
  pool=allowed[rank*40:(rank+1)*40];assert len(pool)==40
  tasks=[]
  for f in sorted(Path(f'/proc/{pid}/task').iterdir()):
   tid=int(f.name)
   try:tasks.append({'tid':tid,'name':(f/'comm').read_text().strip(),'before':sorted(os.sched_getaffinity(tid))})
   except ProcessLookupError:continue
  workers.append({'rank':rank,'pid':pid,'main':pool[2:-2],'acl':[pool[-2]],'release':[pool[-1]],'irq_reserved_unchanged':pool[:2],'tasks':tasks})
 assert sorted(x['rank'] for x in workers)==list(range(16))
 record={'state':'PREPARED','version':version,'parent_model_runtime':parent,'reference':'Pinned Ascend A3 cpu_binding.py GLOBAL_SLICE;640CPU/16NPU,2IRQ reserved,36main,1ACL,1release','scope':'only owned worker/TID affinity; no IRQ writes, no host service changes, no memory migration','workers':workers,'started':time.time()};(dest/'transition.json').write_text(json.dumps(record,indent=2))
 changed=[]
 try:
  for w in workers:
   for task in w['tasks']:
    cpus=w['acl'] if 'acl_thread' in task['name'] else w['release'] if 'release_thread' in task['name'] else w['main']
    try:os.sched_setaffinity(task['tid'],cpus);changed.append(task);task['after']=sorted(os.sched_getaffinity(task['tid']));assert task['after']==cpus
    except ProcessLookupError:task['exited']=True
  record.update(state='APPLIED',finished=time.time());(dest/'transition.json').write_text(json.dumps(record,indent=2))
 except BaseException:
  for task in reversed(changed):
   try:os.sched_setaffinity(task['tid'],task['before'])
   except ProcessLookupError:pass
  record['state']='ROLLED_BACK';(dest/'transition.json').write_text(json.dumps(record,indent=2));raise
 s['cpu_affinity_record']=str(dest/'transition.json');save()
 with (p/'logs/dspark-v8-benchmark-host.log').open('w') as benchmark_log:
  benchmark=subprocess.Popen(['python3',str(p/'tools/native_dspark_v8_benchmark_job.py')],stdout=benchmark_log,stderr=subprocess.STDOUT,start_new_session=True)
 s['benchmark_host_pid']=benchmark.pid;save()
 while worker.poll() is None and api.poll() is None:time.sleep(3)
 s.update(state='FAIL',worker_exit=worker.poll(),api_exit=api.poll(),reason='resident process exited');save()
