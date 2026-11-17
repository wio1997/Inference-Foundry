from pathlib import Path
import datetime,fcntl,hashlib,json,os,socket,time,urllib.request
assert socket.gethostname()=='<ROLE_NODE>'
r=Path('<TASK_ROOT>/sg_pd_g181');old=r/'functional_PD_g182';out=r/'functional_PD_g204';out.mkdir(exist_ok=False)
lock=Path('<COORD_ROOT>/vllm_pd_workload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.monotonic()
def save(name,d):
 p=out/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
def state(phase,**kwargs):save('state.json',{'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,**kwargs})
def http(url,body=None,timeout=20):
 req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as q:assert q.status==200;return q.read()
try:
 original=json.loads((old/'state.json').read_text());assert original['phase']=='failed_requires_review'
 assert not Path('/proc',str(original['pid']),'cmdline').exists(),'Original functional process still live'
 error=json.loads((old/'native_32768_32_error.txt').read_text());assert error['error']['code']=='server_selection_failed'
 audits=[]
 for tag in ['math','tail_127','tail_129','tail_131','tail_8191','tail_8193','openai_short_32']:
  p=old/(tag+'_audit.json');d=json.loads(p.read_text());assert d['passed'] and d['usage']['completion_tokens']>0;audits.append({'audit':d,'original_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'path':str(p)})
 state('waiting_router_healthy_before_new_diagnostic',original_passed=7,original_failure_retained=True)
 seen=[];deadline=time.monotonic()+90
 while len(seen)<2:
  h=json.loads((r/'actual_launch_handle.json').read_text());assert str(r/'node_controller.py').encode() in Path('/proc',str(h['pid']),'cmdline').read_bytes()
  workers=json.loads(http('http://<HOST>:18080/workers',timeout=5));active=[w for w in workers['workers'] if w['is_healthy'] and w['model_id']=='glm53flash' and w['load']==0]
  if {w['worker_type'] for w in active}=={'prefill','decode'}:seen.append(workers)
  else:seen=[]
  assert time.monotonic()<deadline,'Router eligibility not stable; no request submitted'
  if len(seen)<2:time.sleep(5)
 save('router_healthy_before.json',seen)
 # A fresh diagnostic prefix; the original 503 was rejected at worker selection.
 ids=[1200+(i%97) for i in range(32768)]
 body={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':32,'min_new_tokens':32,'ignore_eos':True},'stream':False}
 save('native_32768_32_request.json',body);state('fresh_32768_diagnostic_running',scored=False)
 before=time.monotonic();raw=http('http://<HOST>:18080/generate',body,timeout=300);(out/'native_32768_32_response.json').write_bytes(raw)
 data=json.loads(raw);usage=data['meta_info'];assert usage['prompt_tokens']==32768 and usage['completion_tokens']==32 and usage.get('num_retractions',0)==0
 assert usage['spec_verify_ct']>0 and usage['spec_accept_length']>=1
 current={'case':'native_32768_32_new_prefix_g204','passed':True,'usage':usage,'elapsed_s':time.monotonic()-before,'response_sha256':hashlib.sha256(raw).hexdigest(),'diagnostic_only':True};save('native_32768_32_audit.json',current)
 save('result.json',{'passed':True,'original_G182_failed':True,'original_failure':error,'original_seven':audits,'remaining_new_prefix':current,'service_generation':181,'scope':'Selected realmodel nativePD semantics/length gates; originalfirst503 retained; not all-statebytewise proof, C16 or1024 benchmark, quality suite, orperformance acceptance'})
 for role,url in [('p','http://<HOST>:19083'),('d','http://<HOST>:19084')]:save('after_'+role+'_server_info.json',json.loads(http(url+'/server_info',timeout=20)))
 state('selected_PD_functional_gates_passed_with_original_router_failure_retained',passed_cases=8,scored=False)
except BaseException as exc:state('failed_requires_review',error=repr(exc));raise
