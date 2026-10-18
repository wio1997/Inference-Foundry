"""Short true-PD mode isolation; no performance measurement."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess,time,urllib.request,uuid
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-histogram-live3-20261019/node33')
cid='c7be13e9c4f558e7da2aa395b764cccbd5026bf1dc4f5f3cba52abf4c7dc76db'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
probe=root/'moe-prefix-histogram-probe';assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists()
cases=[z for z in json.loads((root/'prefill-padding-cases-v1.json').read_text())['quality'] if z['case_id']=='quality-0'];assert len(cases)==1
flags=['enable','replica-enable','MTP-only','target-only','diagnose-MTP','diagnose-serial']
out=root/'prefix-quality0-control-v1';out.mkdir(exist_ok=False)
report={'completed':False,'rows':[],'CID':cid,'P_unchanged':'d685575688634c0f6e8878124ceae28b0a48b0def5e73cbafffc5f8ef355d3e4','scope':'Finite ordinary true PD mode isolation. All64 output sequences retained; diagnostic8 requests return native core output. No latency or throughput verdict.'}
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
 rid='prefix3qcontrol-'+mode+'-'+uuid.uuid4().hex
 body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':budget,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':False}
 pbody=dict(body);pbody.update(max_tokens=1,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None})
 pr=post(32,pbody,rid);assert pr['kv_transfer_params']['do_remote_prefill']
 body['kv_transfer_params']=pr['kv_transfer_params'];dr=post(33,body,rid);choice=dr['choices'][0];text=choice['text'];tokens=choice['token_ids'];m=re.search(r'[-+]?\d+(?:\.\d+)?',text)
 row={'mode':mode,'case_id':case['case_id'],'request_id':rid,'budget':budget,'numeric_pass':bool(m and float(m.group())==case['expected']),'full_budget_pass':len(tokens)==budget and dr['usage']['completion_tokens']==budget,'P_response':pr,'D_response':dr,'text':text,'token_ids':tokens};assert row['full_budget_pass'];report['rows'].append(row);idle();(out/'progress.json').write_text(json.dumps(report)+'\n');return row

try:
 for mode,names in [('native-A',[]),('native-replica',['replica-enable']),('both-prefix',['enable']),('native-B',[])]:
  select(names);one(cases[0],mode)
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
