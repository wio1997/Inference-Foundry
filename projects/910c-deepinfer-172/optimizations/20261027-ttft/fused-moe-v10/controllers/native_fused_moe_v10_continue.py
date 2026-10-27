from pathlib import Path
import concurrent.futures,csv,json,subprocess,sys,time,traceback,urllib.request,shutil
p=Path('/data/tiankuan/wio/910c-deepinfer-172');root=p/'benchmarks/fused-moe-v10';status=root/'progress.json';control=p/'serving/fused-moe-v10'
prior=json.loads(status.read_text());assert prior['state']=='FAIL' and len(prior['phases'])==1
assert prior['error'].startswith('AssertionError(')
backup=root/'progress-initial-counter-assertion.json';assert not backup.exists();shutil.copy2(status,backup)
state={k:v for k,v in prior.items() if k not in ['state','phases','error','updated']};state['phases']=[]
state['resume_reason']='Counter check initially omitted eager repair forwards; first raw formal round is retained and independently revalidated, not rerun.'
state['fused_scope']='Eager prefill and eager rejected-verification repair. Captured verification/decode graph continues to use split native MC2.'
def save(value):
 state.update(state=value,updated=time.time());status.write_text(json.dumps(state,indent=2));print(json.dumps({k:v for k,v in state.items() if k!='phases'}),flush=True)
def get(path):
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with op.open('http://127.0.0.1:8172/'+path,timeout=10) as f:return f.read()
def parse_counters(raw):return {line.split()[0]:int(float(line.split()[1])) for line in raw.splitlines() if line.startswith('deepinfer_')}
def counters():
 raw=get('metrics').decode();return parse_counters(raw),raw
def completion(row):
 body={'model':'dsv41','prompt':row['prompt'],'max_tokens':row['max_tokens'],'temperature':0,'ignore_eos':row.get('ignore_eos',False),'stream':False}
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}));req=urllib.request.Request('http://127.0.0.1:8172/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with op.open(req,timeout=900) as f:return json.load(f)
def parallel(rows):
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(completion,rows))
def formal_jobs():
 found=[]
 for d in (control/'jobs').iterdir():
  req=json.loads((d/'request.json').read_text())
  if len(req['requests'])==8 and all(x['max_tokens']==1024 for x in req['requests']):found.append(d)
 return found
try:
 assert json.loads(get('health'))['backend']=='DeepInfer-native'
 for phase in ['measure1','measure2']:
  folder=p/'workload/frozen'/phase;data=[json.loads(line) for line in next(folder.rglob('*.requests.jsonl')).read_text().splitlines()]
  refs=[json.loads(line) for line in (folder/'frozen-vllm-details.jsonl').read_text().splitlines()];by_id={int(row['data_id']):row['prediction'] for row in refs}
  item={'phase':phase};state['phases'].append(item)
  if phase=='measure1':
   item['state']='ADOPTING_EXISTING_FORMAL';save(phase+'_REVALIDATING')
   assert len(formal_jobs())==1
   before=parse_counters((root/(phase+'-metrics-before.txt')).read_text())
   jobs=formal_jobs()
  else:
   assert not list((root/phase).glob('*/performances/*/*.csv'))
   seeds=json.loads((folder/'warmup.json').read_text());item['state']='PREFIX_WARMUP';save(phase+'_PREFIX_WARMUP')
   warmed=parallel([{'prompt':x['prompt'],'max_tokens':1} for x in seeds]);(root/(phase+'-prefix-warmup.json')).write_text(json.dumps(warmed,ensure_ascii=False,indent=2))
   item['state']='C8_WARMUP';save(phase+'_C8_WARMUP')
   warmed=parallel([{'prompt':x['question'],'max_tokens':32,'ignore_eos':True} for x in data]);(root/(phase+'-c8-warmup.json')).write_text(json.dumps(warmed,ensure_ascii=False,indent=2))
   for i,response in enumerate(warmed):
    assert by_id[i].startswith(response['choices'][0]['text']),('warmup mismatch',phase,i)
    assert response['usage']['prompt_tokens']==32768 and response['usage']['completion_tokens']==32
   assert sum(x['usage']['prompt_tokens_details']['cached_tokens'] for x in warmed)==235904
   previous={d.name for d in (control/'jobs').iterdir()};before,raw=counters();(root/(phase+'-metrics-before.txt')).write_text(raw)
   item['state']='AISBENCH_RUNNING';save(phase+'_AISBENCH_RUNNING')
   cmd=[sys.executable,'-m','ais_bench.benchmark.cli.main',str(root/(phase+'.py')),'--mode','perf','--num-warmups','0','--num-prompts','8','--max-num-workers','1']
   with (root/(phase+'-aisbench.log')).open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
   assert result.returncode==0,result.returncode
   jobs=[d for d in (control/'jobs').iterdir() if d.name not in previous];assert len(jobs)==1
  after,raw=counters();(root/(phase+'-metrics-after.txt')).write_text(raw)
  deltas={key:after[key]-before[key] for key in before}
  assert deltas['deepinfer_requests_total']==8 and deltas['deepinfer_prompt_tokens_total']==262144 and deltas['deepinfer_output_tokens_total']==8192 and deltas['deepinfer_cached_tokens_total']==235904 and deltas['deepinfer_failed_requests_total']==0,deltas
  proof=[json.loads(f.read_text()) for f in jobs[0].glob('result-dp*.json')];assert len(proof)==2
  for x in proof:
   st=x['runtime_stats'];repairs=st['prompt_lookup']['repair_forwards']
   assert st['fused_mc2_calls']==40*(4+repairs),(st['fused_mc2_calls'],repairs)
   assert st['native_eager_attention_calls']==0 and st['fused_mhc_layers']==40 and st['mc2_layers']==40
  metrics=[f for f in (root/phase).glob('*/performances/*/*.json') if not f.name.endswith('_details.json')];assert len(metrics)==1
  metric=metrics[0];native=json.loads(metric.read_text());details=[json.loads(x) for x in metric.with_name(metric.stem+'_details.jsonl').read_text().splitlines()]
  assert len(details)==8 and {int(x['data_id']) for x in details}==set(by_id)
  for x in details:
   assert x['success'] and x['input_tokens']==32768 and x['output_tokens']==1024
   assert x['prediction']==by_id[int(x['data_id'])],('full output mismatch',phase,x['data_id'])
  duration=float(str(native['Benchmark Duration']['total']).split()[0])/1000
  table={x['Performance Parameters']:x for x in csv.DictReader(metric.with_suffix('.csv').open())}
  item.update(state='PASS',requests=8,success=8,duration_seconds=duration,output_throughput_tokens_per_second=8192/duration,mean_ttft_ms=float(table['TTFT']['Average'].split()[0]),mean_tpot_ms=float(table['TPOT']['Average'].split()[0]),mean_e2el_ms=float(table['E2EL']['Average'].split()[0]),actual_prefix_hit_rate=deltas['deepinfer_cached_tokens_total']/262144,server_counter_deltas=deltas,exact_full1024_predictions_matching_frozen_vllm=8,fused_path_proof={'job':jobs[0].name,'results':proof},adopted_existing_formal_round=phase=='measure1')
  save('PASS_'+phase)
 state['aggregate_output_throughput_tokens_per_second']=16384/sum(x['duration_seconds'] for x in state['phases'])
 state['mean_ttft_ms']=sum(x['mean_ttft_ms'] for x in state['phases'])/2
 state['performance_goal_met']=state['aggregate_output_throughput_tokens_per_second']>=state['performance_goal_output_tokens_per_second']
 state['ttft_improved']=state['mean_ttft_ms']<4315.8
 state['ttft_reduction_fraction']=1-state['mean_ttft_ms']/4315.8
 state['throughput_change_fraction']=state['aggregate_output_throughput_tokens_per_second']/737.9902137380734-1
 save('COMPLETE' if state['performance_goal_met'] and state['ttft_improved'] else 'PERFORMANCE_GAP')
except BaseException as e:
 state['error']=repr(e);save('FAIL');traceback.print_exc();raise
