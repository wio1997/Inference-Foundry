import pathlib,json,hashlib,math,re
R=pathlib.Path(__file__).resolve().parent;out={'project_id':'glm53-flash','passed':False,'formal_requests':0,'scope':'P2TP8bulk/MTP5 protected fullmodel only; noPD/cache/GraphD/80K/performance acceptance'}
try:
 for f,sha in json.loads((R/'review.json').read_text())['files_sha256'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==sha,f
 h=json.loads(json.loads((R/'node14.json').read_text())['stdout']);r=h['result'];out['generation_requests']=r['generation_requests'];out['HTTP_terminal']=r['HTTP_terminal'];assert r['overlay_file_identity_verified'] and r['ready_ranks']==[0,1] and len(r['requests'])==8
 refs=json.loads((R/'r080-token-reference.json').read_text())['cases'];inputs=json.loads((R/'inputs.json').read_text());out['cases']=[]
 for case in range(4):
  rows=[x for x in r['requests'] if x['case']==case];assert sorted(x['rank'] for x in rows)==[0,1];gold=next(x for x in refs if x['case']==case and x['repeat']==0)
  for row in rows:
   assert row['status']==200;res=row['response'];c=res['choices'][0];lp=c['logprobs'];assert c['token_ids']==gold['token_ids'] and c['text']==gold['text'] and c['finish_reason']==gold['finish_reason'] and c.get('stop_reason')==gold['stop_reason'];assert c['prompt_token_ids']==inputs[case]['input_ids'];assert len(lp['token_logprobs'])==len(gold['token_ids']);assert all(math.isfinite(x) for x in lp['token_logprobs']) and all(math.isfinite(x) for d in lp['top_logprobs'] for x in d.values());delta=max(abs(a-b) for a,b in zip(lp['token_logprobs'],gold['token_logprobs']));assert delta<=.25;assert res['usage']['prompt_tokens']==len(inputs[case]['input_ids']) and res['usage']['completion_tokens']==len(gold['token_ids']);out['cases'].append({'case':case,'rank':row['rank'],'selected_logprob_max_abs':delta,'all_ID_text_EOS_exact':True})
 def counters(text):
  out={}
  for line in text.splitlines():
   m=re.match(r'^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total)(\{[^\n]*\})?\s+(\S+)',line)
   if m:out[m[1]+(m[2] or '')]=float(m[3])
  return out
 out['actual_MTP_counter_deltas']={}
 for rank in ['0','1']:
  before=counters(r['metrics_before'][rank]);after=counters(r['metrics_after'][rank]);out['actual_MTP_counter_deltas'][rank]={k:v-before.get(k,0.) for k,v in after.items()};assert all(v>=0 for v in out['actual_MTP_counter_deltas'][rank].values())
 out['actual_MTP_drafts_observed']=any('num_drafts_total' in k and v>0 for d in out['actual_MTP_counter_deltas'].values() for k,v in d.items());out['actual_MTP_accepted_observed']=any('num_accepted_tokens_total' in k and v>0 for d in out['actual_MTP_counter_deltas'].values() for k,v in d.items());assert out['actual_MTP_drafts_observed'] and out['actual_MTP_accepted_observed'],'NoactualMTPrequestcounterdelta'
 assert r['native_terminal_codes']=={'0':0,'1':0} and r['passed'];assert h['post_inspect']['stdout'].strip()=='false 0' and h['post_npu']['stdout'].count('No running processes found')==8
 out.update(passed=True,verdict='PASS_P2TP8_NATIVE_BULK_FULLMODEL_PROTECTED_NUMERICS_AND_ACTUAL_MTP5_COMPONENT',generation_requests=8,HTTP_terminal=8)
except Exception as e:out.update(error=repr(e),verdict='FAIL_OR_INCOMPLETE_P2TP8_BULK_MTP5_MODEL_COMPONENT')
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
