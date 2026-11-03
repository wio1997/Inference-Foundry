from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'M15-prompt-consumer-presence-and-recorded-guard-scope-v31.json';assert not OUT.exists()
def pin(p,s):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==s;return json.loads(b)
d=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
prompt=[];head=[]
for db in d['databases']:
 for a in db['API_records']:
  text='\n'.join(a['project_frames'])
  if '_get_prompt_logprobs_dict'in text:prompt.append(dict(database_index=db['database_index'],API=a))
  if '/logits_processor.py'in text and any(x in text for x in['/model_runner_v1.py(2635)','/model_runner_v1.py(2646)']):head.append(dict(database_index=db['database_index'],API=a))
fields={'num_prompt_logprobs','prompt_logprobs','prompt_logprobs_requested','use_aux_hidden_state_outputs','speculative_method','method'};observed=[]
def visit(v,path,root):
 if isinstance(v,dict):
  for key,z in v.items():
   pp=path+'.'+str(key)
   if key in fields:
    val=dict(type=type(z).__name__,count=len(z),empty=not z)if isinstance(z,(list,dict))else z
    observed.append(dict(root_key=root,path=pp,field=key,value=val))
   if isinstance(z,(dict,list)):visit(z,pp,root)
 elif isinstance(v,list):
  for j,z in enumerate(v):
   if isinstance(z,(dict,list)):visit(z,path+'[]',root)
for r in prep['prepare_roots']:visit(r.get('metadata'), 'metadata',r['root_key'])
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),prompt_helper_native_API_records=len(prompt),primary_target_head_native_API_records=len(head),primary_target_head_native_chains=len({r['API']['chain_key']for r in head}),recorded_guard_fields=[dict(path=k[0],value=json.loads(k[1]),roots=v)for k,v in Counter((r['path'],json.dumps(r['value'],sort_keys=True))for r in observed).items()],prompt_guard_fields_found=sorted({r['field']for r in observed if'prompt'in r['field']}),prompt_guard_status='UNKNOWN_IF_NOT_RECORDED_NOT_DEFAULT_EMPTY',new_source_model_device_service_health_work=False,overall_complete=False)
result=dict(summary=summary,prompt_native_API_records=prompt,primary_target_head_API_records=head,primary_target_head_fullchains={k:d['canonical_fullchains'][k]for k in{r['API']['chain_key']for r in head}},recorded_guard_fields=observed,limits=['Zeroprompt-helpernativeAPIframesdoesnotprovefunctioninactive:earlyreturn/no-deviceworkispossible.','MissingpromptguardfieldisUNKNOWN,notemptydictionaryorfalse.','Userrequestabsencealoneisnotactualworkerstateattestation.','Primaryheadsourceframesarepositiveroleevidence;typedTASK/COMMjoinsandM16wholecoveragearenotclaimed.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,head_chain_headers=[dict(chain_key=k,API_records=sum(r['API']['chain_key']==k for r in head),project_frames=[v['frame']for v in d['canonical_fullchains'][k]if'/vllm'in v['frame']])for k in{r['API']['chain_key']for r in head}],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
