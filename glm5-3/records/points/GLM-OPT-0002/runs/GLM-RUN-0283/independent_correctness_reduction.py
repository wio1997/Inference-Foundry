from pathlib import Path
import hashlib,json,math,re
repo=Path('/Users/wio/work/Inference-Foundry-glm5-3');r=repo/'glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0283'
read=lambda n:json.loads((r/n).read_text())
assert read('state.json')['status']==read('diagnostic_status.json')['status']=='completed'
assert not any((r/n).exists() for n in ('failure.json','terminal_failure.json','recovery_failure.json'))
plan=read('functional_plan.json');gold=plan['complete_golden'];guards=read('guards_after.json');before=read('guards_before.json')
rows=[];salts=[]
for label in ('correctness_mode0_short','correctness_mode1_short','correctness_mode1_complete','retained_mode_warm'):
 body=read(label+'_request.json');salts.append(body['cache_salt']);assert body['cache_salt']=='GLM-RUN-0283-'+label
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
 if True:
  t=read(label+'_transfer.json');rid=read(label+'_P.raw')['kv_transfer_params']['remote_request_id'];assert t['remote_request_id']==rid
  assert t['all16_native_transfer_success'] and len(t['rows'])==16 and all('KV cache transfer for request '+rid+' took ' in x for x in t['rows'])
  assert {int(re.search(r'local_device_id (\d+)',x).group(1)) for x in t['rows']}==set(range(16))
 rows.append(dict(label=label,exact_tokens=len(ids),prompt_tokens=prompt,finish_reason=reasons[-1],workload_signature=sig,SSE_sha256=hashlib.sha256((r/(label+'_D.sse')).read_bytes()).hexdigest()))
assert len(salts)==len(set(salts))==4
witnesses=[read(n) for n in ('correctness_mode0_witness.json','correctness_mode1_witness.json','complete_witness.json','retained_witness.json')]
assert before['167']['health']==200 and before['167']['idle'] and len(before['167']['device_owners'])==16
assert witnesses[1]['H13_runtime']==witnesses[2]['H13_runtime']
raw_captures={rank:read('witnesses/h13_capture_rank%d.json'%rank) for rank in range(16)}
pids=set(guards['167']['worker_namespace_pids'].values())
assert {z['pid'] for z in raw_captures.values()}==pids
for w,mode,t in zip(witnesses,(0,1,1,0),(1,2,2,3)):
 assert w['H6_mode']==w['H5_event_mode']==1 and w['H11_mode']==w['H9_mode']==0 and w['H13_mode']==mode and w['transition']==t
 assert all(z['dispatch_cached_true'] and z['combine_cached_true'] and z['mode']==1 for z in w['H6']['rows'])
 assert len(w['H5_rows'])==16 and all(z['returned_none'] and z['mode']==1 and not z['configured_overlap'] for z in w['H5_rows'])
 assert len(w['H9_rows'])==16 and all('FULL' in z['runtime_mode'] and z['capture_num_graphs']>=1 and z['num_tokens']==2 and z['eager_breaks']==0 for z in w['H9_rows'])
 assert len(w['H11_rows'])==16 and all(z['mode']==0 and all(not v['actual_replay'] and v['selected_runtime']=='NONE' for v in z['rows']) for z in w['H11_rows'])
 for key in ('H13_capture','H13_runtime'):
  assert len(w[key])==16 and {z['rank'] for z in w[key]}==set(range(16)) and {z['pid'] for z in w[key]}==pids
 for z in w['H13_capture']:
  assert z==raw_captures[z['rank']]
  assert z['raw_model_shared'] and z['independent_pool'] and z['bank_pools'][0]!=z['bank_pools'][1]
  assert z['same_input_addresses'] and z['output_pointers_disjoint']
  assert len(z['entries'])==2 and len(z['entries'][0])==len(z['entries'][1])==1
  a,b=z['entries'][0][0],z['entries'][1][0]
  assert a['descriptor']==b['descriptor'] and a['input_addresses']==b['input_addresses']
  assert a['graph_count']>=1 and b['graph_count']>=1
  assert {o['data_ptr'] for o in a['output']}.isdisjoint({o['data_ptr'] for o in b['output']})
  geometry=[v for v in z['layouts'] if v['fast_gate'] and v['input']['shape']==[2,6144] and v['router_input']['shape']==[2,256] and v['padded_num_tokens']==16 and not v['replace_allreduce'] and v['tp_size']==16]
  assert {v['mode'] for v in geometry}=={0,1}
  for v in geometry:
   assert v['tp_rank']==z['rank'] and v['bytes_equal'] and v['input_unchanged']
   for name,shape,dtype in [('hidden_states',[1,6144],'torch.bfloat16'),('router_logits',[1,256],'torch.float32'),('mc2_mask',[1],'torch.bool')]:
    a,b=v['original'][name],v['candidate'][name]
    assert a['shape']==b['shape']==shape and a['dtype']==b['dtype']==dtype and a['stride']==b['stride'] and a['format']==b['format']==2
 for z in w['H13_runtime']:
  assert z==read('witnesses/h13_runtime_t%d_rank%d.json'%(t,z['rank']))
  c=raw_captures[z['rank']];e=c['entries'][mode][0]
  assert z['mode']==mode and z['transition']==t and z['graph_count']>=1 and z['eager_breaks']==0 and z['no_value_D2H']
  assert z['pool']==c['bank_pools'][mode] and z['descriptor']==e['descriptor'] and z['output']==e['output']
assert all(z['health']==200 and z['idle'] and len(z['device_owners'])==16 for z in guards.values())
assert guards['166']['root']==before['166']['root'] and guards['166']['worker_start_ticks']==before['166']['worker_start_ticks']
stack=read('retained_stack.json');assert stack['diagnostic_completed'] and stack['terminal_verified'] and not stack['recovery_used'] and stack['H6'] and stack['H5'] and stack['target_FULL'] and not stack['H13'] and not stack['H11']
assert read('active_epoch.json')['epoch']=='candidate' and not (r/'epochs/baseline_recovery/launch.json').exists()
proof=dict(passed=True,independent_raw_reduction=True,completed_device_output_and_exact_standard_PD=True,all16_actual_two_target_graph_banks=True,same_capture_inputs_and_distinct_owned_outputs=True,all16_bit_layout_format_input_nonmutation=True,all16_native_transfer=True,unique_salts=salts,request_results=rows,H6_H5_retained=True,target_FULL=True,H11_mode=0,H13_mode=0,terminal_transition=3,terminal_health_idle_all16=True,recovery_used=False,performance_gain=None,Current=None,limitations='Four scoped short/exact natural23 PD correctness requests; no long concurrency/fullAPI/SLA/production two-pool memory proof. Aggregate work not ordered engine trace; no performance conclusion.')
p=r/'correctness_reduced.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='request_results'}))
