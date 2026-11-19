"""C1 raw API A/B; only return_logprob changes. Two exact prior prompts, two repeats.
No NPU observer armed, no service/math/config changes, no retries.
"""
from pathlib import Path
import datetime,hashlib,json,os,signal,time,urllib.request,urllib.error
OUT=Path('/task/client_ds41_g503/diagnostic_logprob_AB_g1115');OUT.mkdir(exist_ok=False);os.chdir(OUT);start=time.monotonic()
def save(name,value):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def state(phase,**more):
 d={'phase':phase,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'elapsed_s':time.monotonic()-start,**more};save('state.json',d);print(json.dumps(d),flush=True)
def alarm(*_):raise TimeoutError('RawAPI logprob AB240s bound; no retry')
signal.signal(signal.SIGALRM,alarm);signal.alarm(240)
cases=json.loads((Path(__file__).parent/'cases.json').read_text());assert len(cases)==2
plan=[{'case':c,'repeat':rep,'logprob':enabled} for rep in range(2) for c in cases for enabled in ([False,True] if rep==0 else [True,False])];assert len(plan)==8
save('manifest.json',{'cases':cases,'plan':plan,'scope':'C1 exactsameAPI/prompts/sampling_params/max128; onlyreturn_logprobchanged, top_logprobs_num0both. ABthenBAorder/noretry. Notfullquality/performance.','concurrency':1,'observer_armed':False})
rows=[]
try:
 for item in plan:
  c=item['case'];enabled=item['logprob'];tag=c['case']+'_r'+str(item['repeat'])+('_on' if enabled else '_off');state('case_running',case=tag)
  body={'input_ids':c['input_ids'],'sampling_params':{'temperature':0,'sampling_seed':0,'max_new_tokens':128},'stream':False,'return_logprob':enabled,'logprob_start_len':-1,'top_logprobs_num':0,'return_text_in_logprobs':True}
  save(tag+'_request.json',body);t=time.monotonic();req=urllib.request.Request('http://<ROLE_HOST>:19094/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=100) as f:assert f.status==200;raw=f.read()
  (OUT/(tag+'_response.json')).write_bytes(raw);x=json.loads(raw);rows.append({'case':c['case'],'repeat':item['repeat'],'logprob':enabled,'tag':tag,'expected':c['expected'],'response':x,'response_SHA256':hashlib.sha256(raw).hexdigest(),'elapsed_s':time.monotonic()-t});save('partial_result.json',rows)
 save('result.json',rows);state('diagnostic_complete_requires_parent_review',completed=len(rows))
except urllib.error.HTTPError as e:
 (OUT/'HTTP_error_body.bin').write_bytes(e.read());state('failed_requires_review',error=repr(e));raise
except BaseException as e:state('failed_requires_review',error=repr(e));raise
