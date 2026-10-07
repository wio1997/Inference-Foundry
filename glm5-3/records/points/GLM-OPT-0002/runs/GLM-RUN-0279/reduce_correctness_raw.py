from pathlib import Path
import hashlib,json,math,re
repo=Path('/Users/wio/work/Inference-Foundry-glm5-3');r=repo/'glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0279'
read=lambda n:json.loads((r/n).read_text())
assert read('state.json')['status']==read('diagnostic_status.json')['status']=='completed'
assert not any((r/n).exists() for n in ('failure.json','terminal_failure.json','recovery_failure.json'))
plan=read('functional_plan.json');gold=plan['complete_golden'];guards=read('guards_after.json');before=read('guards_before.json')
rows=[];salts=[]
for label in ('correctness_mode0_short','correctness_mode1_short','correctness_mode1_complete','retained_mode_warm'):
 body=read(label+'_request.json');salts.append(body['cache_salt']);assert body['cache_salt']=='GLM-RUN-0279-'+label
 raw=(r/(label+'_D.sse')).read_text();assert raw.rstrip().endswith('data: [DONE]')
 sse=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line!='data: [DONE]'];assert all('error' not in e for e in sse)
 events=read(label+'_events.json');assert [e['value'] for e in events if e['value'] is not None]==sse
 ids=[];content='';reasons=[];usage=[]
 for e in sse:
  if e.get('usage'):usage.append(e['usage'])
  for c in e.get('choices',[]):
   ids+=c.get('token_ids') or c.get('delta',{}).get('token_ids') or [];content+=c.get('delta',{}).get('content') or ''
   if c.get('finish_reason'):reasons.append(c['finish_reason'])
 complete=label.endswith('_complete');expected=gold['token_ids'] if complete else [785,1196,374,10156,264,3405,304,8452];prompt=gold['prompt_tokens'] if complete else 2334
 result=read(label+'_result.json');assert ids==expected==result['token_ids'] and usage[-1]==dict(prompt_tokens=prompt,completion_tokens=len(ids),total_tokens=prompt+len(ids))
 assert reasons[-1]==result['finish_reason']==('stop' if complete else 'length')
 if complete:assert content==gold['final_content'] and result['semantic_accepted']
 w=read(label+'_workload.json');b=w['before']['counters'];a=w['after']['counters'];assert set(a)==set(b)
 delta={k:a[k]-b[k] for k in b};assert delta==w['delta'] and all(math.isfinite(v) and v>=0 and v.is_integer() for v in delta.values())
 counts=[int(next(v for k,v in delta.items() if k.startswith('vllm:spec_decode_num_'+n+'_total{'))) for n in ('drafts','draft_tokens','accepted_tokens','accepted_tokens_per_pos')]
 assert 0<=counts[2]==counts[3]<=counts[1]<=counts[0]
 sig=dict(num_drafts=counts[0],num_draft_tokens=counts[1],num_accepted_tokens=counts[2],accepted_position0=counts[3],invalid_draft_tokens=counts[0]-counts[1]);assert w['signature']==sig
 external=result['external_KV_delta'];assert next(v for k,v in external.items() if '_queries_total{' in k)==prompt and next(v for k,v in external.items() if '_hits_total{' in k)==prompt
 if label!='retained_mode_warm':
  t=read(label+'_transfer.json');rid=read(label+'_P.raw')['kv_transfer_params']['remote_request_id'];assert t['remote_request_id']==rid
  assert t['all16_native_transfer_success'] and len(t['rows'])==16 and all('KV cache transfer for request '+rid+' took ' in x for x in t['rows'])
  assert {int(re.search(r'local_device_id (\d+)',x).group(1)) for x in t['rows']}==set(range(16))
 rows.append(dict(label=label,exact_tokens=len(ids),prompt_tokens=prompt,finish_reason=reasons[-1],workload_signature=sig,SSE_sha256=hashlib.sha256((r/(label+'_D.sse')).read_bytes()).hexdigest()))
assert len(salts)==len(set(salts))==4
witnesses=[read(n) for n in ('correctness_mode0_witness.json','correctness_mode1_witness.json','complete_witness.json','retained_witness.json')]
assert witnesses[1]['H12_rows']==witnesses[2]['H12_rows']
for w,mode in zip(witnesses,(0,1,1,0)):
 assert w['H6_mode']==w['H5_event_mode']==1 and w['H11_mode']==w['H9_mode']==0 and w['H12_mode']==mode
 assert all(z['dispatch_cached_true'] and z['combine_cached_true'] and z['mode']==1 for z in w['H6']['rows'])
 assert len(w['H5_rows'])==16 and all(z['returned_none'] and z['mode']==1 and not z['configured_overlap'] for z in w['H5_rows'])
 assert len(w['H9_rows'])==16 and all('FULL' in z['runtime_mode'] and z['capture_num_graphs']>=1 and z['num_tokens']==2 and z['eager_breaks']==0 for z in w['H9_rows'])
 assert len(w['H11_rows'])==16 and all(z['mode']==0 and all(not v['actual_replay'] and v['selected_runtime']=='NONE' for v in z['rows']) for z in w['H11_rows'])
 layouts=w['H12_rows'];assert len(layouts)==16 and {z['rank'] for z in layouts}==set(range(16));assert {z['pid'] for z in layouts}==set(guards['167']['worker_namespace_pids'].values())
 for z in layouts:
  assert z['source_sha256']==plan['rotary_shim_sha256'] and z['mode']==mode and z['tables_established_by_registration'] and z['no_D2H_values']
  assert len(z['rows'])==4 and all(v['use_cache'] for v in z['rows'])
  for key,stride in (('baseline',[128,1]),('candidate',[64,1])):
   assert len(z[key])==2 and all(t['shape']==[1048576,64] and t['stride']==stride and t['dtype']=='torch.bfloat16' and t['format']==2 for t in z[key])
  assert all(v['selected']==z['baseline' if mode==0 else 'candidate'] for v in z['rows'])
  assert all(all(t['shape']==[2,1,1,64] for t in v['output']) for v in z['rows'])
  assert all(all(t['data_ptr']==p['data_ptr'] for t,p in zip(v['output'],v['persistent'])) for v in z['rows'] if v['use_cache'])
for rank in range(16):
 zs=[next(z for z in w['H12_rows'] if z['rank']==rank) for w in witnesses]
 assert [z['transition'] for z in zs]==[1,2,2,3]
 for z in zs[1:]:
  assert z['baseline']==zs[0]['baseline'] and z['candidate']==zs[0]['candidate']
  assert [v['persistent'] for v in z['rows']]==[v['persistent'] for v in zs[0]['rows']]
assert all(z['health']==200 and z['idle'] and len(z['device_owners'])==16 for z in guards.values())
assert guards['166']['root']==before['166']['root'] and guards['166']['worker_start_ticks']==before['166']['worker_start_ticks']
stack=read('retained_stack.json');assert stack['diagnostic_completed'] and stack['terminal_verified'] and not stack['recovery_used'] and stack['H6'] and stack['H5'] and stack['target_FULL'] and not stack['H12'] and not stack['H11']
assert read('active_epoch.json')['epoch']=='candidate' and not (r/'epochs/baseline_recovery/launch.json').exists()
proof=dict(passed=True,independent_raw_reduction=True,completed_device_output_and_exact_standard_PD=True,all16_layout_witness=True,persistent_graph_outputs_unchanged=True,both_tables_lifetime_stable=True,all16_native_transfer=True,unique_salts=salts,request_results=rows,H6_H5_retained=True,target_FULL=True,H11_mode=0,H12_mode=0,terminal_transition=3,terminal_health_idle_all16=True,recovery_used=False,performance_gain=None,Current=None,limitations='Four scoped short/exact natural23 PD correctness requests, no MTP graph/long concurrency/fullAPI/SLA proof. Aggregate work not ordered engine trace; no performance conclusion.')
p=r/'correctness_reduced.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='request_results'}))
