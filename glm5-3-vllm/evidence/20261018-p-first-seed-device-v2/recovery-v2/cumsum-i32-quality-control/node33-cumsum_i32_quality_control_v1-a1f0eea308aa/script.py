"""Short true-PD mode isolation; no performance measurement."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess,time,urllib.request,uuid
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-cumsum-i32-live1-20261019/node33')
cid='416f0dc10e34a115e2537cb62e46e2eb98d30825642ab4ba36d098c072cf11f5'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
probe=root/'moe-cumsum-i32-probe';assert (probe/'target-only').exists();assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists()
cases=[z for z in json.loads((root/'prefill-padding-cases-v1.json').read_text())['quality'] if z['case_id'] in ['quality-0','quality-4','quality-5','quality-13','quality-18','quality-20','quality-21']];assert len(cases)==7
flags=['enable','replica-enable','MTP-only','diagnose-MTP','diagnose-serial']
out=root/'cumsum-i32-quality-control-v1';out.mkdir(exist_ok=False)
report={'completed':False,'rows':[],'CID':cid,'P_unchanged':'13c520313d3369bafc2db811dca35bc93afdfc2ae7ed3572fc1518366c38aceb','scope':'Finite ordinary true PD mode isolation. All64 output sequences retained; MTP stays native in every arm; no intermediate tensor observers enabled. No latency or throughput verdict.'}
def get_metrics(n):
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,29481 if n==32 else 29482),timeout=5) as f:s=f.read().decode()
 return [z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
def idle():
 for i in range(121):
  data={str(n):get_metrics(n) for n in [32,33]}
  if all(len(v)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in v) for v in data.values()):return data
  time.sleep(.25)
 raise AssertionError(data)
def select(names,observe=False):
 idle()
 for flag in flags:(probe/flag).unlink(missing_ok=True)
 for flag in names:(probe/flag).touch()
 if observe:(probe/'disabled').unlink(missing_ok=True)
 else:(probe/'disabled').touch()
def post(n,body,rid):
 req=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(n,29481 if n==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
 with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)
def one(case,mode,budget=64):
 rid='i32qcontrol-'+mode+'-'+uuid.uuid4().hex
 body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':budget,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':False}
 pbody=dict(body);pbody.update(max_tokens=1,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None})
 pr=post(32,pbody,rid);assert pr['kv_transfer_params']['do_remote_prefill']
 body['kv_transfer_params']=pr['kv_transfer_params'];dr=post(33,body,rid);choice=dr['choices'][0];text=choice['text'];tokens=choice['token_ids'];m=re.search(r'[-+]?\d+(?:\.\d+)?',text)
 row={'mode':mode,'case_id':case['case_id'],'request_id':rid,'budget':budget,'numeric_pass':bool(m and float(m.group())==case['expected']),'full_budget_pass':len(tokens)==budget and dr['usage']['completion_tokens']==budget,'P_response':pr,'D_response':dr,'text':text,'token_ids':tokens};assert row['full_budget_pass'];report['rows'].append(row);idle();(out/'progress.json').write_text(json.dumps(report)+'\n');return row

try:
 for case in cases:
  for mode,names in [('native-A',[]),('native-replica',['replica-enable']),('target-i32',['enable']),('native-B',[])]:
   select(names);one(case,mode)
 report['per_case']={case['case_id']:{'unique_sequences':len(set(tuple(z['token_ids']) for z in report['rows'] if z['case_id']==case['case_id'])),'native_A_B_same':next(z['token_ids'] for z in report['rows'] if z['case_id']==case['case_id'] and z['mode']=='native-A')==next(z['token_ids'] for z in report['rows'] if z['case_id']==case['case_id'] and z['mode']=='native-B')} for case in cases}
 report['summary']={'numeric':sum(z['numeric_pass'] for z in report['rows']),'full64':sum(z['full_budget_pass'] for z in report['rows'])}
 report['unique_full64_sequences']=len(set(tuple(z['token_ids']) for z in report['rows']))
 report['after_idle']=idle();report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
finally:
 for flag in flags:(probe/flag).unlink(missing_ok=True)
 (probe/'disabled').touch()
report['candidate_and_diagnostic_markers_removed']=all(not(probe/f).exists() for f in flags)
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'],report.get('failure')
