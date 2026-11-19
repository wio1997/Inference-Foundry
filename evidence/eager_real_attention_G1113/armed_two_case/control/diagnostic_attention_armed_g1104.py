"""Two-case direct raw-generate diagnostic, exact recorded chat prompt IDs.
No model or numerical changes. Native JSON/logprob IDs retained before analysis.
"""
from pathlib import Path
import datetime,hashlib,json,os,signal,time,urllib.request,urllib.error
OUT=Path('/task/client_ds41_g503/diagnostic_attention_armed_g1104');OUT.mkdir(exist_ok=False)
os.chdir(OUT);start=time.monotonic()
def save(name,value):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def state(phase,**kw):
 d={'phase':phase,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'elapsed_s':time.monotonic()-start,**kw};save('state.json',d);print(json.dumps(d),flush=True)
def alarm(*_):raise TimeoutError('Raw generate diagnostic240s bound; no retry')
signal.signal(signal.SIGALRM,alarm);signal.alarm(240)
cases=json.loads((Path(__file__).parent/'cases.json').read_text());assert len(cases)==2
save('manifest.json',{'cases':cases,'scope':'C1 four-case raw generation/token-ID isolation; sameactualSG/native032chatpromptIDs. Notfullquality/performance.','concurrency':1,'retry':False,'max_new_tokens':128})
try:
 rows=[]
 for c in cases:
  tag=c['case'];state('case_running',case=tag)
  body={'input_ids':c['input_ids'],'sampling_params':{'temperature':0,'sampling_seed':0,'max_new_tokens':128},'stream':False,'return_logprob':True,'logprob_start_len':-1,'top_logprobs_num':5,'return_text_in_logprobs':True}
  save(tag+'_request.json',body);t=time.monotonic();req=urllib.request.Request('http://<ROLE_HOST>:19094/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=100) as f:assert f.status==200;raw=f.read()
  (OUT/(tag+'_response.json')).write_bytes(raw);value=json.loads(raw);rows.append({'case':tag,'expected':c['expected'],'response':value,'elapsed_s':time.monotonic()-t,'response_SHA256':hashlib.sha256(raw).hexdigest()});save('partial_result.json',rows)
 save('result.json',rows);state('diagnostic_complete_requires_parent_review',completed=len(rows))
except urllib.error.HTTPError as e:
 (OUT/'HTTP_error_body.bin').write_bytes(e.read());state('failed_requires_review',error=repr(e));raise
except BaseException as e:state('failed_requires_review',error=repr(e));raise
