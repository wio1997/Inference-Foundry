"""Selected natural-text/usage gates for the dedicated native profiling epoch."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,socket,subprocess,time,urllib.request
ROOT=Path('/PRIVATE_TASK_ROOT')
RUN=ROOT/'vllm_pd_g727';OUT=RUN/'functional_g788'
P='http://<ROLE_HOST>:19103';D='http://<ROLE_HOST>:19104';PROXY='http://<ROLE_HOST>:18100'
PCID='d30b701d751751b522af6e472b982eabb198550234c086871176e368170c7b70'
assert socket.gethostname()=='<ROLE_HOSTNAME>'
lock=Path('/root/.codex/coord/deepseek-v41-p3d4-workload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
OUT.mkdir(exist_ok=False);began=time.monotonic()
def save(name,value):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def stamp(phase,**more):
 v=dict(phase=phase,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-began,**more);save('state.json',v);print(json.dumps(v),flush=True)
def http(url,body=None,timeout=20):
 req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as f:assert f.status==200;return f.read()
def owned():
 s=json.loads((RUN/'state.json').read_text());assert s['generation']==727 and s['container_id']==PCID
 assert str(RUN/'node_controller.py').encode() in Path('/proc',str(s['pid']),'cmdline').read_bytes()
 c=json.loads(subprocess.check_output(['docker','inspect',PCID]))[0]
 assert c['State']['Running'] and c['Image']=='sha256:d5492ecfdb3071ff2d4bae3f849245014c54b63035d9eefb35243869461ddae1' and c['Config']['Labels']['codex.run']==RUN.name
 links=[str(p.resolve()) for p in Path('/proc',str(s['pid']),'fd').iterdir()]
 for i in range(8):
  key='<ROLE_HOSTNAME>-npu'+str(i);assert '/root/.codex/coord/'+key+'.lock' in links
  owner=json.loads(Path('/root/.codex/coord',key+'.owner.json').read_text());assert owner['pid']==s['pid'] and owner['state']=='claimed' and owner['run_directory']==str(RUN)
 return s,c
def metrics(tag):
 result={}
 for role,url in [('prefill',P),('decode',D)]:
  text=http(url+'/metrics').decode();(OUT/(tag+'_'+role+'_metrics.txt')).write_text(text)
  counts={}
  for key in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:prompt_tokens_total','vllm:generation_tokens_total']:
   values=[float(x) for x in re.findall(r'^'+re.escape(key)+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',text,re.M)];assert values,(role,key);counts[key]=sum(values)
  assert counts['vllm:num_requests_running']==counts['vllm:num_requests_waiting']==0
  result[role]=counts
 save(tag+'_counters.json',result);return result
try:
 stamp('waiting_existing_727_pair',ready_bound_s=1200)
 deadline=time.monotonic()+1200
 while True:
  s,c=owned()
  try:
   for url in [P,D]:http(url+'/health',timeout=3)
   break
  except Exception:
   assert time.monotonic()<deadline,'Existing727notready; preserve state, no restart'
   time.sleep(10)
 for role,url in [('prefill',P),('decode',D)]:
  model=json.loads(http(url+'/v1/models'));assert model['data'][0]['id']=='ds41flash';save(role+'_models.json',model)
  schema=json.loads(http(url+'/openapi.json'));assert all(p in schema['paths'] for p in ['/start_profile','/stop_profile'])
 native='/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py'
 digest=subprocess.check_output(['docker','exec',PCID,'sha256sum',native],text=True).split()[0]
 assert digest=='ffd9a54456dc6ad10bc7668bc2db52c81edfd5a477f707b41a8be33f56353208'
 assert not (RUN/'proxy_actual_handle_g788.json').exists()
 with socket.socket() as sock:sock.bind(('0.0.0.0',18100))
 remaining=int(72000-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(json.loads((RUN/'actual_launch_handle.json').read_text())['at'])).total_seconds())-120;assert remaining>7200
 argv=['flock','--nonblock','/root/.codex/coord/<ROLE_HOSTNAME>-port18100.lock','docker','exec','-w','/task/vllm_pd_g727',PCID,'timeout','--signal=TERM','--kill-after=30',str(remaining),'python3',native,'--host','0.0.0.0','--port','18100','--prefiller-hosts','<ROLE_HOST>','--prefiller-ports','19103','--decoder-hosts','<ROLE_HOST>','--decoder-ports','19104','--max-retries','1','--max-waiting-retries','1','--workers','1']
 with (RUN/'proxy_g788.log').open('x') as log:proxy=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (RUN/'proxy_actual_handle_g788.json').write_text(json.dumps(dict(pid=proxy.pid,argv=argv,container_id=PCID,source_SHA256=digest,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),bound_s=remaining),indent=2)+'\n')
 deadline=time.monotonic()+120
 while True:
  assert proxy.poll() is None,'Nativeproxyexited; inspectexistinglog'
  try:health=json.loads(http(PROXY+'/healthcheck',timeout=3));save('proxy_healthcheck.json',health);break
  except Exception:assert time.monotonic()<deadline;time.sleep(2)
 before=metrics('before');stamp('selected_natural_text_requests_running')
 for tag,prompt,answer in [('math','计算 2+2。最终答案只写一个数字。','4'),('capital','What is the capital of France? Reply with only the city name.','Paris')]:
  body={'model':'ds41flash','messages':[{'role':'user','content':prompt}],'temperature':0,'seed':0,'max_tokens':512,'stream':False,'chat_template_kwargs':{'thinking':False}};save(tag+'_request.json',body)
  raw=http(PROXY+'/v1/chat/completions',body,timeout=180);(OUT/(tag+'_response.json')).write_bytes(raw);response=json.loads(raw)
  text=response['choices'][0]['message'].get('content') or '';final=text.split('</think>')[-1].strip().rstrip('。.!！ ');assert final==answer,(tag,final)
  assert response['usage']['completion_tokens']>0;save(tag+'_audit.json',{'passed':True,'usage':response['usage'],'response_sha256':hashlib.sha256(raw).hexdigest()})
 after=metrics('after_semantic')
 text='Native profiling epoch protocol check. Explain how a matrix multiplication combines rows and columns.'
 tokenized=json.loads(http(P+'/tokenize',{'model':'ds41flash','prompt':text}));expected=tokenized['count'];assert expected>0
 body={'model':'ds41flash','prompt':text,'temperature':0,'seed':0,'max_tokens':1024,'min_tokens':1024,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}};save('usage_request.json',body)
 raw=http(PROXY+'/v1/completions',body,timeout=180);(OUT/'usage_response.sse').write_bytes(raw)
 events=[line[5:].strip() for line in raw.decode().splitlines() if line.startswith('data:')];assert events[-1]=='[DONE]'
 objects=[json.loads(line) for line in events if line!='[DONE]'];usage=[x['usage'] for x in objects if x.get('usage')][-1];assert usage['prompt_tokens']==expected and usage['completion_tokens']==1024
 final=metrics('after_usage');delta={role:{k:final[role][k]-after[role][k] for k in final[role]} for role in final}
 assert delta['decode']['vllm:generation_tokens_total']==1024 and delta['prefill']['vllm:generation_tokens_total']==1,delta
 summary={'passed':True,'epoch':727,'selected_semantic_cases':2,'usage':usage,'independent_counter_deltas':delta,'SSE_DONE':True,'scope':'Profile-only CPUannotation runtime selectedgate; no fullquality/performance/allranktrace acceptance. Original630 fullGSMquality remains separate evidence.'}
 save('result.json',summary);stamp('selected_functional_and_usage_passed_requires_parent_review',**summary)
except BaseException as error:
 stamp('failed_requires_review',error=repr(error));raise
