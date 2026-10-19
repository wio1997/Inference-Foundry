"""Actual74400/1 P-prefix warm and alternate-DP cache-read phases, no baseline."""
import hashlib,json,struct,sys,time,urllib.request,urllib.error
from pathlib import Path
RUN=Path('/run/glm53');data=Path('/dataset');ids=json.loads((data/'prefix-00.json').read_text())['input_ids']
assert len(ids)==74400 and hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()=='b49ef8d1a93d8229930061a30ae5b1fe327fc3249a35e42c699fed0043ddbbba'
NS='glm53-prefix-cache-v3-20261017';phase=sys.argv[1];assert phase in ['warm','read']
specs=[('warm-P0',0,NS)] if phase=='warm' else [('read-P1',1,NS),('cold-control-P1',1,NS+'-cold-control')]
report={'project_id':'glm53','namespace':NS,'phase':phase,'functional_only':True,'formal_requests':0,'cases':[]}
def metrics(label):
 with urllib.request.urlopen('http://172.16.12.30:30300/metrics',timeout=5) as r:raw=r.read()
 name=f'metrics-{label}-node30.txt';(RUN/name).write_bytes(raw);return {'file':name,'sha256':hashlib.sha256(raw).hexdigest()}
report['metrics_before']=metrics(phase+'-before')
try:
 for label,rank,salt in specs:
  rid=NS+'-'+label;row={'case':label,'P_global_dp':rank,'P_wire_local_dp':rank,'request_id':rid,'namespace':salt,'required_input':74400,'required_output':1,'input_sha256_le_u32':'b49ef8d1a93d8229930061a30ae5b1fe327fc3249a35e42c699fed0043ddbbba','dispatch_monotonic_ns':time.monotonic_ns()};report['cases'].append(row)
  body=dict(model='glm53-w8a8',prompt=ids,max_tokens=1,min_tokens=1,ignore_eos=True,temperature=0,seed=20261017,stream=False,request_id=rid,cache_salt=salt,return_token_ids=False,skip_special_tokens=False,logprobs=5)
  try:
   req=urllib.request.Request('http://172.16.12.30:30300/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-data-parallel-rank':str(rank),'X-Request-Id':rid})
   with urllib.request.urlopen(req,timeout=240) as response:raw=response.read();row['http_status']=response.status
   row['response']=json.loads(raw);u=row['response']['usage'];row['pass']=bool(u['prompt_tokens']==74400 and u['completion_tokens']==1 and row['response']['choices'][0]['finish_reason']=='length')
  except urllib.error.HTTPError as e:row.update(error=str(e),http_status=e.code,upstream_body=e.read().decode(),**{'pass':False})
  except Exception as e:row.update(error=repr(e),**{'pass':False})
  finally:row['terminal_monotonic_ns']=time.monotonic_ns();row['metrics']=metrics(label)
  print('GLM53_PREFIX_CASE='+json.dumps(row),flush=True)
  (RUN/f'prefix-{phase}-progress.json').write_text(json.dumps(report,indent=2)+'\n')
  assert row['pass'],'Actual prefix request failed, no retry'
finally:
 (RUN/f'prefix-{phase}-result.json').write_text(json.dumps(report,indent=2)+'\n')
 print('GLM53_PREFIX_RESULT='+json.dumps(report),flush=True)
