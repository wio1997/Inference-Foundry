"""Two serial first-token observations on identical explicit30 input token IDs.
Not full quality or performance. No retries and no concurrent requests.
"""
from pathlib import Path
import datetime,hashlib,json,os,signal,time,urllib.request
ROOT=Path('/task/client_ds41_g1122');OUT=ROOT/'same_prefix_unarmed_g1148';OUT.mkdir(exist_ok=False);start=time.monotonic()
IDS=[0, 128803, 15434, 345, 926, 4306, 8834, 15, 2616, 16, 60919, 223, 929, 2734, 223, 24, 14, 1539, 1258, 223, 25, 16, 19414, 418, 1353, 270, 9132, 16, 128804, 128822]
def save(name,value):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def state(phase,**more):save('state.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,pid=os.getpid(),elapsed_s=time.monotonic()-start,**more))
def alarm(*_):raise TimeoutError('Sameprefix diagnostic hard600s bound, no retry')
signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
requests=[('native','http://<ROLE_HOST>:19103/v1/completions',{'model':'ds41flash','prompt':IDS,'max_tokens':1,'temperature':0,'seed':0,'stream':False,'logprobs':20}),('SG','http://<ROLE_HOST>:19094/generate',{'input_ids':IDS,'sampling_params':{'max_new_tokens':1,'temperature':0},'sampling_seed':0,'return_logprob':True,'top_logprobs_num':20})]
save('manifest.json',{'case':'math1','expected_integer':'91','input_ids':IDS,'real_rows':30,'first_generated_position':30,'request_order':['native','SG'],'observer_ARM':False,'NoSP':True,'eager':True,'concurrency':1,'retries':0,'scope':'Unarmedobservercontrol, explicitsameprefix firsttoken comparison; not semanticcohort/fullquality/performance.'});results=[]
try:
 for role,url,body in requests:
  state('request_inflight',role=role);save(role+'_request.json',body);t=time.monotonic()
  req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=280) as f:b=f.read();assert f.status==200
  (OUT/(role+'_response.json')).write_bytes(b);response=json.loads(b)
  if role=='native':assert response['usage']['prompt_tokens']==30 and response['usage']['completion_tokens']==1
  else:assert response['meta_info']['prompt_tokens']==30 and response['meta_info']['completion_tokens']==1
  row={'role':role,'transport_ok':True,'elapsed_s':time.monotonic()-t,'raw_response_SHA256':hashlib.sha256(b).hexdigest(),'response':response};results.append(row);save('partial_result.json',results)
 save('result.json',results);state('complete_requires_parent_review',completed=2)
except BaseException as e:
 save('partial_result.json',results);state('failed_requires_review',error=repr(e));raise
