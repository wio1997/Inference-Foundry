from pathlib import Path
import concurrent.futures,csv,hashlib,json,os,subprocess,sys,time,traceback,urllib.request
p=Path('/data/tiankuan/wio/910c-deepinfer-172');root=p/'benchmarks/fused-moe-v10';status=root/'progress.json';assert not status.exists()
state={'state':'STARTING','backend':'DeepInfer native W8A8','started':time.time(),'phases':[],
 'formal_workload':{'input_tokens':32768,'output_tokens':1024,'concurrency':8,'rounds':2,'target_actual_cached_tokens_per_round':235904},
 'performance_goal_output_tokens_per_second':701.0907030511697, 'reference_native_v9c_mean_ttft_ms':4315.8}
def save(value):
 state['state']=value;state['updated']=time.time();status.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
def get(path):
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with op.open('http://127.0.0.1:8172/'+path,timeout=10) as r:return r.read()
def counters():
 raw=get('metrics').decode();return {line.split()[0]:int(float(line.split()[1])) for line in raw.splitlines() if line.startswith('deepinfer_')},raw
def completion(request):
 body={'model':'dsv41','prompt':request['prompt'],'max_tokens':request['max_tokens'],'temperature':0,'ignore_eos':request.get('ignore_eos',False),'stream':False}
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 req=urllib.request.Request('http://127.0.0.1:8172/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with op.open(req,timeout=900) as r:return json.load(r)
def parallel(rows):
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(completion,rows))
try:
 for attempt in range(120):
  try:
   health=json.loads(get('health'));assert health['backend']=='DeepInfer-native';break
  except Exception:time.sleep(1)
 else:raise RuntimeError('native service not ready')
 for phase in ['measure1','measure2']:
  folder=p/'workload/frozen'/phase
  data=[json.loads(line) for line in next(folder.rglob('*.requests.jsonl')).open()]
  refs=[json.loads(line) for line in (folder/'frozen-vllm-details.jsonl').open()]
  by_id={int(row['data_id']):row['prediction'] for row in refs}
  seeds=json.loads((folder/'warmup.json').read_text())
  item={'phase':phase,'state':'PREFIX_WARMUP'};state['phases'].append(item);save(phase+'_PREFIX_WARMUP')
  warmed=parallel([{'prompt':row['prompt'],'max_tokens':1} for row in seeds])
  (root/(phase+'-prefix-warmup.json')).write_text(json.dumps(warmed,ensure_ascii=False,indent=2))
  item['state']='C8_WARMUP';save(phase+'_C8_WARMUP')
  warmed=parallel([{'prompt':row['question'],'max_tokens':32,'ignore_eos':True} for row in data])
  (root/(phase+'-c8-warmup.json')).write_text(json.dumps(warmed,ensure_ascii=False,indent=2))
  for i,response in enumerate(warmed):
   text=response['choices'][0]['text'];usage=response['usage']
   assert by_id[i].startswith(text),('canonical output mismatch during required C8 warmup',phase,i,text,by_id[i][:len(text)])
   assert usage['prompt_tokens']==32768 and usage['completion_tokens']==32,usage
  assert sum(r['usage']['prompt_tokens_details']['cached_tokens'] for r in warmed)==235904
  before_jobs={d.name for d in (p/'serving/fused-moe-v10/jobs').iterdir()}
  before,raw_before=counters();(root/(phase+'-metrics-before.txt')).write_text(raw_before)
  item['state']='AISBENCH_RUNNING';save(phase+'_AISBENCH_RUNNING')
  command=[sys.executable,'-m','ais_bench.benchmark.cli.main',str(root/(phase+'.py')),'--mode','perf','--num-warmups','0','--num-prompts','8','--max-num-workers','1']
  with (root/(phase+'-aisbench.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,('AISBench exit',r.returncode)
  after_jobs=[d for d in (p/'serving/fused-moe-v10/jobs').iterdir() if d.name not in before_jobs]
  assert len(after_jobs)==1,after_jobs
  proof=[json.loads(f.read_text()) for f in after_jobs[0].glob('result-dp*.json')]
  assert len(proof)==2 and all(x['runtime_stats']['fused_mc2_calls']==160 and x['runtime_stats']['native_eager_attention_calls']==0 for x in proof),proof
  item['fused_path_proof']={'job':after_jobs[0].name,'results':proof}
  after,raw_after=counters();(root/(phase+'-metrics-after.txt')).write_text(raw_after)
  metrics=[f for f in (root/phase).glob('*/performances/*/*.json') if not f.name.endswith('_details.json')]
  assert len(metrics)==1,metrics
  metric_file=metrics[0];native=json.loads(metric_file.read_text())
  details=[json.loads(line) for line in metric_file.with_name(metric_file.stem+'_details.jsonl').open()]
  assert len(details)==8 and all(row['success'] and row['input_tokens']==32768 and row['output_tokens']==1024 for row in details)
  for row in details:assert row['prediction']==by_id[int(row['data_id'])],('canonical full1024 output mismatch',phase,row['data_id'])
  deltas={key:after[key]-before[key] for key in before}
  assert deltas['deepinfer_requests_total']==8 and deltas['deepinfer_prompt_tokens_total']==262144 and deltas['deepinfer_output_tokens_total']==8192 and deltas['deepinfer_cached_tokens_total']==235904,deltas
  def val(key):return float(str(native[key]['total']).split()[0])
  duration=val('Benchmark Duration')/1000
  table={row['Performance Parameters']:row for row in csv.DictReader(metric_file.with_suffix('.csv').open())}
  item.update(state='PASS',requests=8,success=8,duration_seconds=duration,
   output_throughput_tokens_per_second=8192/duration,mean_ttft_ms=float(table['TTFT']['Average'].split()[0]),
   mean_tpot_ms=float(table['TPOT']['Average'].split()[0]),mean_e2el_ms=float(table['E2EL']['Average'].split()[0]),
   actual_prefix_hit_rate=deltas['deepinfer_cached_tokens_total']/deltas['deepinfer_prompt_tokens_total'],
   server_counter_deltas=deltas,exact_full1024_predictions_matching_frozen_vllm=8)
  save('PASS_'+phase)
 state['aggregate_output_throughput_tokens_per_second']=16384/sum(row['duration_seconds'] for row in state['phases'])
 state['performance_goal_met']=state['aggregate_output_throughput_tokens_per_second']>=state['performance_goal_output_tokens_per_second']
 state['mean_ttft_ms']=sum(row['mean_ttft_ms'] for row in state['phases'])/2
 state['ttft_improved']=state['mean_ttft_ms']<4315.8
 state['ttft_reduction_fraction']=1-state['mean_ttft_ms']/4315.8
 state['throughput_change_fraction']=state['aggregate_output_throughput_tokens_per_second']/737.9902137380734-1
 save('COMPLETE' if state['performance_goal_met'] and state['ttft_improved'] else 'PERFORMANCE_GAP')
except BaseException as exc:
 state['error']=repr(exc);save('FAIL');traceback.print_exc();raise
