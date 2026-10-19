from pathlib import Path
import collections,itertools,json
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';reports=[]
for label,key in [('c4-control-logprobs','D_chunks'),('native-D-c4','chunks')]:
 j=json.loads(next((e/label).rglob('metrics.json')).read_text());rows=[];violations=[]
 for bidx,burst in enumerate(j['bursts']):
  for r in burst['rows']:
   flat=[]
   for ch in r[key]:
    for c in ch.get('choices',[]):
     ids=c.get('token_ids') or [];lp=c.get('logprobs')
     if not ids:continue
     assert lp and len(lp['tokens'])==len(ids) and lp['tokens']==['token_id:'+str(x) for x in ids]
     for i,tok in enumerate(ids):
      tops=lp['top_logprobs'][i];chosen=lp['token_logprobs'][i];selected=lp['tokens'][i];assert abs(tops[selected]-chosen)<1e-6
      if chosen<max(tops.values())-1e-6:violations.append({'id':r['header_request_id'],'position':len(flat),'selected':selected,'top5':tops})
      flat.append({'selected_id':tok,'selected_logprob':chosen,'top5':tops})
   assert len(flat)==64
   body=r.get('D_request',r.get('request'));rows.append({'burst':bidx,'N':r['input_tokens'],'header_id':r['header_request_id'],'prompt_ids':body['prompt'],'ids':r['token_ids'],'flat':flat,'numeric':r['numeric_answer_correct']})
 examples=[]
 for n in (256,257):
  rs=[r for r in rows if r['N']==n]
  pairs=[]
  for a,b in itertools.combinations(rs,2):
   if a['ids']==b['ids']:continue
   k=next(i for i,(x,y) in enumerate(zip(a['ids'],b['ids'])) if x!=y);assert a['prompt_ids']==b['prompt_ids'] and a['ids'][:k]==b['ids'][:k]
   aa,bb=a['flat'][k],b['flat'][k];ka,kb='token_id:'+str(a['ids'][k]),'token_id:'+str(b['ids'][k]);both=all(z in q['top5'] for z in (ka,kb) for q in (aa,bb))
   pairs.append({'N':n,'first_difference_index0':k,'shared_output_prefix_ids':a['ids'][:k],'a_header':a['header_id'],'b_header':b['header_id'],'a':aa,'b':bb,'both_candidates_in_both_top5':both,'relative_a_minus_b':{side:q['top5'][ka]-q['top5'][kb] for side,q in [('a',aa),('b',bb)]} if both else None})
  pairs.sort(key=lambda p:(not p['both_candidates_in_both_top5'],p['first_difference_index0']))
  if pairs:examples.append(pairs[0])
 report={'label':label,'requests':len(rows),'numeric_correct':sum(r['numeric'] for r in rows),'unique64_by_N':{n:len({tuple(r['ids']) for r in rows if r['N']==n}) for n in (256,257)},'selected_token_not_argmax_top5_count':len(violations),'violations':violations[:10],'same_reported_history_divergence_examples':examples,'scope':'API-reported same logical input/history and returned target logprobs. No independent GPU KV/input equality or raw logit dtype proof. Native D variation without PD demonstrates that PD transfer is not necessary for observed variation; it does not exclude an additional PD effect or establish harmless numerical noise.'};reports.append(report)
(e/'c4-logprob-diagnosis.json').write_text(json.dumps(reports,indent=2)+'\n')
for r in reports:print(json.dumps(r))
