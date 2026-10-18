from pathlib import Path
import concurrent.futures,datetime,json,re,socket,subprocess,time,urllib.request,uuid
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live2-20261018/node33');cid='05e98c8425d8bb0db0c30425ae054f9e8ee7b747914a18defaa2f18d8aee8f2a';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];plan=json.loads((root/'plan.json').read_text());assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];formal=json.loads((root/'sfa-replay-order-c4-balanced1/result.json').read_text());assert formal['completed'] and formal['numeric_all_pass'];assert (root/'sfa-replay-order-probe/disabled').exists();marker=root/'sfa-replay-order-probe/enable';assert not marker.exists();cases=json.loads((root/'prefill-padding-cases-v1.json').read_text());out=root/'sfa-replay-order-cancel-v1';out.mkdir(exist_ok=False);started=datetime.datetime.now(datetime.timezone.utc);report={'completed':False,'rows':[],'scope':'DirectnativePremoteDecode->D128, clientcloseafterfirsttoken, immediatecold64reuse; thenfourconcurrent128 withoneclose andthreecomplete peers per mode. Trackactualbackendabort evidence separately from clientclose. No performanceclaim.'}
def metrics(node):
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,29481 if node==32 else 29482),timeout=5) as f:s=f.read().decode()
 return {z.split(' ',1)[0]:float(z.rsplit(' ',1)[1]) for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{','vllm:request_success_total{','vllm:generation_tokens_total{'))}
def idle():
 for i in range(121):
  values={str(n):metrics(n) for n in [32,33]}
  selected=[v for d in values.values() for k,v in d.items() if k.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
  assert len(selected)>=6
  if all(v==0 for v in selected):return {'polls':i+1,'metrics':values}
  time.sleep(.25)
 raise AssertionError(values)
def post(node,body,rid):
 req=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(node,29481 if node==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid});return urllib.request.urlopen(req,timeout=120)
def one(case,mode,phase,close=False,budget=128):
 rid='cancel-sfa-'+uuid.uuid4().hex;body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':budget,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':True,'stream_options':{'include_usage':True}};pbody={k:v for k,v in body.items() if k!='stream_options'};pbody.update(max_tokens=1,stream=False,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None});row={'request_id':rid,'case':case['case_id'],'mode':mode,'phase':phase,'close_requested':close,'budget':budget,'request':body,'tokens':[],'text':'','chunks':[],'sent_ns':time.monotonic_ns()};report['rows'].append(row)
 with post(32,pbody,rid) as f:row['P_response']=json.load(f)
 body['kv_transfer_params']=row['P_response']['kv_transfer_params'];response=post(33,body,rid)
 try:
  row['status']=response.status
  for line in response:
   if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
   z=json.loads(line[6:]);assert not z.get('error'),z;row['chunks'].append(z)
   for c in z.get('choices',[]):row['tokens']+=c.get('token_ids') or [];row['text']+=c.get('text','')
   if close and row['tokens']:row['closed_after_token_count']=len(row['tokens']);row['close_ns']=time.monotonic_ns();break
 finally:response.close()
 row['finished_ns']=time.monotonic_ns();m=re.search(r'[-+]?\d+(?:\.\d+)?',row['text']);assert row['status']==200 and m and float(m.group())==case['expected'],row['text']
 if close:assert 0<len(row['tokens'])<budget
 else:
  usage=next(z['usage'] for z in reversed(row['chunks']) if z.get('usage'));assert len(row['tokens'])==budget and usage['completion_tokens']==budget and usage['prompt_tokens']==len(case['prompt_token_ids']);row['usage']=usage;row['full_response_pass']=True
 return row
try:
 report['before']=idle()
 for mode in ['native','adaptive']:
  idle()
  if mode=='adaptive':marker.touch()
  else:marker.unlink(missing_ok=True)
  for i in range(4):
   row=one(cases['fixed'][i%2],mode,'sequential-close',True);row['idle_after_close']=idle();one(cases['fixed'][(i+1)%2],mode,'immediate-cold-reuse',False,64);idle();(out/'progress.json').write_text(json.dumps(report)+'\n')
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   jobs=[pool.submit(one,case,mode,'C4-one-close-three-peers',index==0,128) for index,case in enumerate(cases['mixed'])]
   for job in jobs:job.result()
  idle();one(cases['fixed'][0],mode,'post-C4-immediate-cold-reuse',False,64);idle()
 report['after']=idle();q=subprocess.run(['docker','logs','--timestamps','--since',started.isoformat(),cid],capture_output=True,text=True,timeout=20);own=[z['request_id'] for z in report['rows'] if z['close_requested']];lines=(q.stdout+'\n'+q.stderr).splitlines();matched={rid:[s for s in lines if rid in s and any(w in s.lower() for w in ['abort','cancel'])] for rid in own};report['backend_abort_log_evidence']=matched;report['backend_abort_log_request_count']=sum(bool(v) for v in matched.values());before=report['before']['metrics']['33'];after=report['after']['metrics']['33'];report['abort_counter_deltas']={k:v-before.get(k,0) for k,v in after.items() if 'abort' in k.lower()};report['client_close_requests']=len(own);report['full_reuse_and_peer_responses']=sum(z.get('full_response_pass',False) for z in report['rows']);report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
finally:marker.unlink(missing_ok=True)
(out/'result.json').write_text(json.dumps(report)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')
