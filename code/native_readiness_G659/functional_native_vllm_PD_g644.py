"""Bounded native0.32 PD protocol/selected semantic and length diagnostics."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,signal,socket,subprocess,time,urllib.error,urllib.request
ROOT=Path('/PRIVATE_TASK_ROOT');RUN=ROOT/'vllm_pd_g630';OUT=RUN/'functional_PD_g644';P='http://<ROLE_HOST>:19103';D='http://<ROLE_HOST>:19104';PROXY='http://<ROLE_HOST>:18100'
os.chdir(ROOT);assert socket.gethostname()=='<ROLE_HOSTNAME>'
lock=Path('/root/.codex/coord/deepseek-v41-p3d4-workload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False);started=time.monotonic()
def save(name,data):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def state(phase,**kw):
 d=dict(phase=phase,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,**kw);save('state.json',d);print(json.dumps(d),flush=True)
def alarm(*_):raise TimeoutError('G644total10800s bound; no service restart')
signal.signal(signal.SIGALRM,alarm);signal.alarm(10800)
def http(url,body=None,timeout=15):
 req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as f:assert f.status==200;return f.read()
def own_P():
 s=json.loads((RUN/'state.json').read_text());assert s['generation']==630 and s['phase']=='model_starting';p=Path('/proc',str(s['pid']),'cmdline');assert p.exists() and str(RUN/'node_controller.py').encode() in p.read_bytes();a=json.loads(subprocess.check_output(['docker','inspect',s['container_id']],text=True,cwd=ROOT))[0];assert a['State']['Running'] and a['Image']=='sha256:d5492ecfdb3071ff2d4bae3f849245014c54b63035d9eefb35243869461ddae1' and a['Config']['Labels']['codex.run']==RUN.name and a['Id']==s['container_id'];return s,a
try:
 state('waiting_specific_native_G630_pair',ready_wait_s=2400)
 deadline=time.monotonic()+2400
 while True:
  s,a=own_P()
  try:
   for b in [P,D]:http(b+'/health',timeout=3)
   break
  except (urllib.error.URLError,TimeoutError,socket.timeout):
   assert time.monotonic()<deadline,'Actual engines not ready inside functional observation bound; do not restart them'
   time.sleep(10)
 ids={}
 for role,b in [('p',P),('d',D)]:
  models=json.loads(http(b+'/v1/models'));save('startup_'+role+'_models.json',models);ids[role]=models['data'][0]['id']
 assert ids['p']==ids['d'];served=ids['p']
 native='/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py';expected='ffd9a54456dc6ad10bc7668bc2db52c81edfd5a477f707b41a8be33f56353208'
 actual=subprocess.check_output(['docker','exec',a['Id'],'sha256sum',native],text=True,cwd=ROOT).split()[0];assert actual==expected,('Nativeproxy source differs from exact0.32commit',actual)
 proxy_handle=json.loads((RUN/'proxy_actual_handle_g640.json').read_text());p=Path('/proc',str(proxy_handle['pid']),'cmdline');assert p.exists() and b'18100' in p.read_bytes() and a['Id'].encode() in p.read_bytes();assert proxy_handle['source_SHA256']==expected and proxy_handle['container_id']==a['Id']
 save('adopt_existing_native_proxy.json',proxy_handle)
 http(PROXY+'/healthcheck',timeout=8);models=json.loads(http(PROXY+'/v1/models',timeout=8));assert models['data'][0]['id']==served
 save('proxy_healthcheck_before.json',json.loads(http(PROXY+'/healthcheck')));http(PROXY+'/healthcheck',timeout=8);state('native_proxy_ready_no_generation_claim')
 audits=[]
 def case(tag,body,length=None,count=32,answer=None,chat=False):
  assert time.monotonic()-started<10500;own_P();http(P+'/health');http(D+'/health');body['model']=served
  for role,b in [('p',P),('d',D)]:
   (OUT/(tag+'_'+role+'_metrics_before.txt')).write_bytes(http(b+'/metrics'))
  save(tag+'_request.json',body);state('request_running',case=tag,HTTP_bound_s=1200);before=time.monotonic()
  try:raw=http(PROXY+('/v1/chat/completions' if chat else '/v1/completions'),body,timeout=1200)
  except urllib.error.HTTPError as exc:(OUT/(tag+'_error.txt')).write_bytes(exc.read());raise
  (OUT/(tag+'_response.json')).write_bytes(raw);r=json.loads(raw);u=r.get('usage') or {};assert u.get('completion_tokens',0)>0,(tag,u)
  if count is not None:assert u['completion_tokens']==count,(tag,u)
  if length is not None:assert u['prompt_tokens']==length,(tag,u)
  if answer is not None:
   text=r['choices'][0]['message'].get('content') or '';final=text.split('</think>')[-1].strip().rstrip('。.!！ ');assert final==answer,(tag,final)
  audit={'case':tag,'passed':True,'usage':u,'response_sha256':hashlib.sha256(raw).hexdigest(),'elapsed_s':time.monotonic()-before,'scope':'Selected nativePD generation/protocol only, no qualitysuite or performance acceptance'};save(tag+'_audit.json',audit);audits.append(audit);state('case_passed',case=tag)
  for role,b in [('p',P),('d',D)]:(OUT/(tag+'_'+role+'_metrics_after.txt')).write_bytes(http(b+'/metrics'))
  return r
 case('math',{'messages':[{'role':'user','content':'计算 2+2。最终答案只写一个数字。'}],'temperature':0,'max_tokens':512,'stream':False,'chat_template_kwargs':{'thinking':False}},count=None,answer='4',chat=True)
 for length in [8191,8193,127,129,131]:case('tail_'+str(length),{'prompt':[1000+(i%100) for i in range(length)],'temperature':0,'max_tokens':32,'min_tokens':32,'ignore_eos':True,'stream':False},length=length)
 case('openai_short_32',{'prompt':'Explain the role of matrix multiplication in neural networks.','temperature':0,'max_tokens':32,'min_tokens':32,'ignore_eos':True,'stream':False})
 case('native_32768_32',{'prompt':[1000+(i%100) for i in range(32768)],'temperature':0,'max_tokens':32,'min_tokens':32,'ignore_eos':True,'stream':False},length=32768)
 for length in [129,8193,32768]:
  original='native_32768_32' if length==32768 else 'tail_'+str(length);old=json.loads((OUT/(original+'_response.json')).read_text());body=json.loads((OUT/(original+'_request.json')).read_text());tag='cache_replay_'+str(length);new=case(tag,body,length=length)
  assert old['choices'][0]['text']==new['choices'][0]['text'],(tag,'Greedy text changed on sameinput')
  cached=(new['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0);assert cached>=128,(tag,new['usage']);save(tag+'_coherence_audit.json',{'passed':True,'same_greedy_text':True,'reported_P_cached_tokens':cached,'scope':'Nativeproxy reports P cached count fromactualprefillusage; does not independently prove Dcachehit/allcomponentbytes'})
 for role,b in [('p',P),('d',D)]:save('after_'+role+'_models.json',json.loads(http(b+'/v1/models')))
 save('result.json',{'passed':True,'cases':audits,'scope':'Selected fullnativePD generation+semantic/length/greedyPprefixreplay only. Fullweightcoverage, alltarget/draftgraphreplay, directstatebytecoherence, qualitysuite, C16 andperformance still requireindependentevidence'})
 state('selected_functional_cases_passed_requires_parent_and_graph_PD_review')
except BaseException as exc:state('failed_requires_review',error=repr(exc));raise
