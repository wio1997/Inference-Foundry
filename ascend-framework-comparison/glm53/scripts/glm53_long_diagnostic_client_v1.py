"""Bounded actual long-context first-token/logprob diagnostic; no baseline."""
import hashlib,json,struct,time,urllib.request
from pathlib import Path
RUN=Path('/run/glm53'); DATA=Path('/dataset'); NS='glm53-long-diagnostic-20261017-v1'
ids=json.loads((DATA/'request-p00-v1.json').read_text())['input_ids']
assert len(ids)==80000
assert hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()=='43a309afc4f0652d712dfa3c5318735b680a685ba179d49539c9a1d952a98529'
report={'project_id':'glm53','diagnostic_only':True,'cases':[],'model_http_attempts':0}
def call(role,label,salt,count,params=None):
 node,port=(30,30300) if role=='P' else (43,30400)
 rid=NS+'-p00-'+label+'.'+role
 body=dict(model='glm53-w8a8',prompt=ids,max_tokens=count,min_tokens=count,temperature=0,seed=20261017,ignore_eos=True,stream=False,cache_salt=salt,request_id=rid,return_token_ids=True,skip_special_tokens=False,logprobs=5)
 if params:body['kv_transfer_params']=params
 row={'role':role,'request_id':rid,'namespace':salt,'required_input':80000,'required_output':count,'dispatch_monotonic_ns':time.monotonic_ns()};report['model_http_attempts']+=1
 try:
  req=urllib.request.Request(f'http://172.16.12.{node}:{port}/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-data-parallel-rank':'0','X-Request-Id':rid})
  with urllib.request.urlopen(req,timeout=900) as response:raw=response.read();row['http_status']=response.status
  data=json.loads(raw);prompt=data.pop('prompt_token_ids',None)
  if prompt is not None:
   row['echoed_prompt_count']=len(prompt);row['echoed_prompt_equal']=prompt==ids
   assert prompt==ids
  row['response']=data;row['wire_response_sha256']=hashlib.sha256(raw).hexdigest()
  assert data['usage']['prompt_tokens']==80000 and data['usage']['completion_tokens']==count
  assert data['choices'][0]['finish_reason']=='length'
  row['pass']=True
 except Exception as error:
  row['error']=repr(error);row['pass']=False
 finally:row['terminal_monotonic_ns']=time.monotonic_ns()
 return row
try:
 specs=[('direct-D-a','D',NS+'-dcold-a'),('direct-D-b','D',NS+'-dcold-b'),('direct-P-a','P',NS+'-pcold-a'),('direct-P-b','P',NS+'-pcold-b'),('cold-PD-a','PD',NS+'-pdcold-a'),('cold-PD-b','PD',NS+'-pdcold-b'),('reused-PD','PD','glm53-long-functional-20261017-v1-warm'),('reused-D','D','glm53-long-functional-20261017-v1-warm')]
 for label,mode,salt in specs:
  item={'case':label,'namespace':salt,'rows':[]};report['cases'].append(item)
  if mode=='PD':
   pr=call('P',label,salt,1,{'do_remote_decode':True,'do_remote_prefill':False});item['rows'].append(pr)
   assert pr['pass']
   params=pr['response']['kv_transfer_params']
   assert params['remote_num_prompt_tokens']==80000 and params['remote_host']=='172.16.12.30' and params['remote_port']==26032
   item['rows'].append(call('D',label,salt,8,params))
  else:item['rows'].append(call(mode,label,salt,8))
  (RUN/'diagnostic-progress.json').write_text(json.dumps(report,indent=2)+'\n')
  print('GLM53_DIAGNOSTIC_CASE='+json.dumps(item),flush=True)
  assert all(r['pass'] for r in item['rows']), 'HTTP/usage failure; preserve and stop'
finally:
 (RUN/'diagnostic-result.json').write_text(json.dumps(report,indent=2)+'\n')
 print('GLM53_DIAGNOSTIC_RESULT='+json.dumps(report),flush=True)
