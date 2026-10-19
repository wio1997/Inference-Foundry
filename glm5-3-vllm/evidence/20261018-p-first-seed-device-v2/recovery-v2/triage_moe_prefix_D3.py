"""Short true-PD mode isolation; no performance measurement."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess,time,urllib.request,uuid
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-histogram-live3-20261019/node33')
cid='c7be13e9c4f558e7da2aa395b764cccbd5026bf1dc4f5f3cba52abf4c7dc76db'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
probe=root/'moe-prefix-histogram-probe';assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists()
cases=json.loads((root/'prefill-padding-cases-v1.json').read_text())['fixed']
flags=['enable','replica-enable','MTP-only','target-only','diagnose-MTP','diagnose-serial']
out=root/'prefix-mode-triage-v1';out.mkdir(exist_ok=False)
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
 rid='prefix3diag-'+mode+'-'+uuid.uuid4().hex
 body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':budget,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':False}
 pbody=dict(body);pbody.update(max_tokens=1,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None})
 pr=post(32,pbody,rid);assert pr['kv_transfer_params']['do_remote_prefill']
 body['kv_transfer_params']=pr['kv_transfer_params'];dr=post(33,body,rid);choice=dr['choices'][0];text=choice['text'];tokens=choice['token_ids'];m=re.search(r'[-+]?\d+(?:\.\d+)?',text)
 row={'mode':mode,'case_id':case['case_id'],'request_id':rid,'budget':budget,'numeric_pass':bool(m and float(m.group())==case['expected']),'full_budget_pass':len(tokens)==budget and dr['usage']['completion_tokens']==budget,'P_response':pr,'D_response':dr,'text':text,'token_ids':tokens};assert row['full_budget_pass'];report['rows'].append(row);idle();(out/'progress.json').write_text(json.dumps(report)+'\n');return row

def diagnostics():
 files=[];rows=[]
 for p in sorted(probe.glob('diagnostic-rank-*.jsonl')):
  b=p.read_bytes();files.append({'name':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});rows.extend(json.loads(s) for s in b.decode().splitlines())
 return {'files':files,'rows':rows,'ranks':sorted(set(z['rank'] for z in rows))}
try:
 select([],True)
 for c in cases:one(c,'native-before')
 report['MTP_core_comparison_reused_from_D2']='32 actual pairs bitwise; D3 changes target family capture lifecycle, not native MTP core.'
 for mode,names in [('native-replica',['replica-enable']),('MTP-only',['enable','MTP-only']),('target-only',['enable','target-only']),('both-prefix',['enable']),('native-after',[])]:
  select(names,True)
  for c in cases:one(c,mode)
 report['summary']={mode:{'requests':len(v),'numeric':sum(z['numeric_pass'] for z in v),'full_budget':sum(z['full_budget_pass'] for z in v)} for mode in sorted(set(z['mode'] for z in report['rows'])) for v in [[z for z in report['rows'] if z['mode']==mode]]}
 reference={z['case_id']:z['token_ids'] for z in report['rows'] if z['mode']=='native-before'}
 report['full64_equal_to_native_before']=[{'mode':z['mode'],'case_id':z['case_id'],'equal':z['token_ids']==reference[z['case_id']]} for z in report['rows'] if z['budget']==64]
 inventory={};bindings=[];raw=[]
 for p in sorted(probe.glob('rank-*.jsonl')):
  data=p.read_bytes();raw.append({'name':p.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  for line in data.decode().splitlines():
   z=json.loads(line)
   for g in z['graphs']:
    inventory.setdefault(str(z['rank']),g)
    if 'runtime_input_inventory' not in g:continue
    runtime=g['runtime_input_inventory'];captures=g['capture_input_inventories'];diff={}
    for family,items in captures.items():
     keys=sorted(set(runtime)|set(items));diff[family]=[{'path':k,'capture':items.get(k),'runtime':runtime.get(k)} for k in keys if runtime.get(k)!=items.get(k)]
    bindings.append({'rank':z['rank'],'selected_family':g['selected_family'],'N':g['tokens'],'runtime_entries':len(runtime),'differing_capture_bindings':diff,'capture_input_inventories':captures,'runtime_input_inventory':runtime})
 report['capture_inventory_by_rank']=inventory;report['binding_diagnostics']=bindings;report['raw_path_files']=raw
 report['after_idle']=idle();report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
finally:
 for flag in flags:(probe/flag).unlink(missing_ok=True)
 (probe/'disabled').touch()
report['candidate_and_diagnostic_markers_removed']=all(not (probe/f).exists() for f in flags)
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'],report.get('failure')
