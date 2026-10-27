from pathlib import Path
import json,subprocess,time,urllib.request,os
p=Path('/data/tiankuan/wio/910c-deepinfer-172');v='prefill-tokenizer-v21';parent='prefill-tail-ffn-v20';control=p/'serving'/parent;source=p/'src/DeepInfer-v21-tokenizer'
status=p/'runs'/('serving-'+v+'-status.json');assert not status.exists()
old=json.loads((p/'runs'/('serving-'+parent+'-status.json')).read_text())
worker_pid=old['worker_host_pid'];os.kill(worker_pid,0)
env='PYTHONPATH='+str(source)+':/usr/local/Ascend/cann-9.2.0/python/site-packages:/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe'
api_cmd=['docker','exec','-e',env,'-w',str(source),'wio-deepinfer-s172','python','examples/ascend910c_w8a8_api.py','--model-dir','/models/DeepSeek-V4.1-Flash-1017-w8a8','--control-dir',str(control),'--tokenizer-cache-tokens','1048576']
s={'state':'HTTP_STARTING','started':time.time(),'source_manifest':'evidence/'+v+'-source/SHA256.json','worker_command':old['worker_command'],'worker_host_pid':worker_pid,'api_command':api_cmd,'actual_control_dir':str(control),'model_runtime_unchanged':parent,'cpu_affinity_record':old['cpu_affinity_record'],'tokenizer_cache':{'token_budget':1048576,'max_entries':64,'scope':'exact text to input IDs only; model outputs always recomputed'},'model_source_unchanged':'src/DeepInfer-v20-tailffn'}
def save():status.write_text(json.dumps(s,indent=2))
with (p/'logs'/('serving-'+v+'-api.log')).open('w') as log:
 api=subprocess.Popen(api_cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);s['api_host_pid']=api.pid;save()
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for _ in range(60):
  try:
   with op.open('http://127.0.0.1:8172/health',timeout=2) as f:health=json.load(f)
   assert health=={'status':'ok','backend':'DeepInfer-native'};s.update(state='READY',health=health);save();break
  except Exception:
   if api.poll() is not None:break
   time.sleep(1)
 else:raise RuntimeError('API readiness timeout')
 assert s['state']=='READY','API failed readiness'
 with (p/'logs'/('prefill-tokenizer-v21-benchmark-host.log')).open('w') as log:
  benchmark=subprocess.Popen(['python3',str(p/'tools/native_prefill_tokenizer_v21_benchmark_job.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 s['benchmark_host_pid']=benchmark.pid;save()
 while api.poll() is None:
  os.kill(worker_pid,0);time.sleep(3)
 s.update(state='FAIL',reason='API process exited',api_exit=api.poll());save()
