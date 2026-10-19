"""Two actual4096/1 directP cases isolate native MemCache read liveness."""
import hashlib,json,struct,time,urllib.request,urllib.error
from pathlib import Path
RUN=Path('/run/glm53');root=Path('/dataset');ids=json.loads((root/'prefix-00.json').read_text())['input_ids']
assert len(ids)==74400 and hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()=='b49ef8d1a93d8229930061a30ae5b1fe327fc3249a35e42c699fed0043ddbbba'
ids=ids[:4096];report={'project_id':'glm53','diagnostic_only':True,'formal_requests':0,'cases':[]}
def metrics(label):
 with urllib.request.urlopen('http://172.16.12.30:30300/metrics',timeout=5) as r:data=r.read()
 (RUN/f'metrics-{label}-node30.txt').write_bytes(data)
 return {'sha256':hashlib.sha256(data).hexdigest(),'file':f'metrics-{label}-node30.txt'}
report['before_metrics']=metrics('before')
try:
 for label,salt in [('known-prefix-P-dp1','glm53-long-functional-20261017-v1-warm'),('cold-control-P-dp1','glm53-cache-size-diagnostic-20261017-v1-cold')]:
  rid='glm53-cache-size-diagnostic-20261017-v1-'+label
  row={'case':label,'request_id':rid,'namespace':salt,'input_sha256_le_u32':hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest(),'required_input':4096,'required_output':1,'P_global_dp':1,'P_wire_local_dp':1,'dispatch_monotonic_ns':time.monotonic_ns()}
  report['cases'].append(row)
  body=dict(model='glm53-w8a8',prompt=ids,max_tokens=1,min_tokens=1,ignore_eos=True,temperature=0,seed=20261017,stream=False,request_id=rid,cache_salt=salt,return_token_ids=False,skip_special_tokens=False,logprobs=5)
  try:
   req=urllib.request.Request('http://172.16.12.30:30300/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-data-parallel-rank':'1','X-Request-Id':rid})
   with urllib.request.urlopen(req,timeout=120) as response:raw=response.read();row['http_status']=response.status
   row['response']=json.loads(raw);usage=row['response']['usage'];row['pass']=bool(usage['prompt_tokens']==4096 and usage['completion_tokens']==1 and row['response']['choices'][0]['finish_reason']=='length')
  except urllib.error.HTTPError as error:row.update(http_status=error.code,error=str(error),upstream_body=error.read().decode(),**{'pass':False})
  except Exception as error:row.update(error=repr(error),**{'pass':False})
  finally:row['terminal_monotonic_ns']=time.monotonic_ns();row['metrics']=metrics(label)
  (RUN/'cache-diagnostic-progress.json').write_text(json.dumps(report,indent=2)+'\n')
  print('GLM53_CACHE_DIAGNOSTIC_CASE='+json.dumps(row),flush=True)
finally:
 (RUN/'cache-diagnostic-result.json').write_text(json.dumps(report,indent=2)+'\n')
 print('GLM53_CACHE_DIAGNOSTIC_RESULT='+json.dumps(report),flush=True)
assert len(report['cases'])==2 and all(c['pass'] for c in report['cases'])
