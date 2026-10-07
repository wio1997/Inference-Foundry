from pathlib import Path
import json,hashlib,re,math
r=Path('/Users/wio/work/Inference-Foundry-glm5-3/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0280');read=lambda n:json.loads((r/n).read_text())
assert read('state.json')['status']=='completed' and read('comparison_status.json')['status']=='completed'
assert not (r/'failure.json').exists() and not (r/'failure_recovery.json').exists()
p=r.parent/'GLM-RUN-0279';gold=json.loads((p/'functional_plan.json').read_text())['complete_golden'];guards=read('guards_after.json');before=read('guards_before.json');rows=[];salts=[]
labels=['priming_layout_warm','A1_warm','A1_complete','B1_warm','B1_complete','A2_warm','A2_complete','B2_warm','B2_complete','retained_mode_warm']
for label in labels:
 complete=label.endswith('_complete');result=read(label+'_matched_result.json');body=read(label+'_request.json');salts.append(body['cache_salt']);assert body['cache_salt']=='GLM-RUN-0280-'+label
 raw=(r/(label+'_D.sse')).read_text();assert raw.rstrip().endswith('data: [DONE]')
 sse=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line!='data: [DONE]'];events=read(label+'_events.json');assert [z['value'] for z in events if z['value'] is not None]==sse
 assert all('error' not in z for z in sse);ids=[];content='';usage=[];reasons=[];arrivals=[]
 for event in events:
  z=event['value']
  if z is None:continue
  if z.get('usage'):usage.append(z['usage'])
  for c in z.get('choices',[]):
   tokens=c.get('token_ids') or c.get('delta',{}).get('token_ids') or []
   if tokens:arrivals.append(event['arrived_ns'])
   ids+=tokens;content+=c.get('delta',{}).get('content') or ''
   if c.get('finish_reason'):reasons.append(c['finish_reason'])
 expected=gold['token_ids'] if complete else [785,1196,374,10156,264,3405,304,8452];prompt=58 if complete else 2334
 assert ids==expected==result['token_ids'] and usage[-1]==dict(prompt_tokens=prompt,total_tokens=prompt+len(ids),completion_tokens=len(ids))
 assert reasons[-1]==result['finish_reason']==('stop' if complete else 'length')
 if complete:assert content==gold['final_content'] and result['semantic_accepted']
 tpot=(arrivals[-1]-arrivals[0])/1e6/(len(ids)-1);assert abs(tpot-result['TPOT_ms'])<1e-9
 workload=read(label+'_workload.json');b=workload['before']['counters'];a=workload['after']['counters'];assert set(b)==set(a)
 delta={k:a[k]-b[k] for k in b};assert delta==workload['delta'] and all(math.isfinite(v) and v>=0 and v.is_integer() for v in delta.values())
 counts=[int(next(v for k,v in delta.items() if k.startswith('vllm:spec_decode_num_'+n+'_total{'))) for n in ('drafts','draft_tokens','accepted_tokens','accepted_tokens_per_pos')]
 assert 0<=counts[2]==counts[3]<=counts[1]<=counts[0]
 expected_work=dict(num_drafts=counts[0],num_draft_tokens=counts[1],num_accepted_tokens=counts[2],accepted_position0=counts[3],invalid_draft_tokens=counts[0]-counts[1]);assert workload['signature']==result['workload_signature']==expected_work
 cb=read(label+'_cache_before.json');ca=read(label+'_cache_after.json');cache_rows={}
 for pod in ('172.16.10.166:9081','172.16.10.167:9900'):
  assert set(cb[pod])==set(ca[pod])=={'0'};values={k:ca[pod]['0'][k]-cb[pod]['0'][k] for k in cb[pod]['0']};assert values=={k:result['cache_by_endpoint'][pod][k] for k in values}
  assert values==dict(hbm_queries=prompt,hbm_hits=0,ext_queries=prompt,ext_hits=0 if pod.endswith('9081') else prompt);cache_rows[pod]=values
 transfer=read(label+'_transfer.json');assert transfer['all16_native_transfer_success'] and len(transfer['rows'])==16
 assert {int(re.search(r'local_device_id (\d+)',s).group(1)) for s in transfer['rows']}==set(range(16));assert read(label+'_P.raw')['kv_transfer_params']['remote_request_id']==transfer['remote_request_id']
 assert all(math.isfinite(result[k]) and result[k]>0 for k in ('D_wall_s','PD_wall_s','P_wall_s'))
 rows.append(dict(label=label,exact_tokens=len(ids),finish_reason=reasons[-1],TPOT_ms=tpot,D_wall_s=result['D_wall_s'],PD_wall_s=result['PD_wall_s'],P_wall_s=result['P_wall_s'],workload_signature=expected_work,cache=cache_rows,SSE_sha256=hashlib.sha256((r/(label+'_D.sse')).read_bytes()).hexdigest()))
assert len(salts)==len(set(salts))==10
initial=json.loads((p/'correctness_reduced.json').read_text())['terminal_transition']
witnesses=[]
for label,mode,t in [('priming_layout',1,initial+1),('A1_warm',0,initial+2),('B1_warm',1,initial+3),('A2_warm',0,initial+4),('B2_warm',1,initial+5)]:
 w=read(label+'_witness.json');layouts=w['H12_rows'];witnesses.append(w)
 assert w['H6_mode']==w['H5_event_mode']==1 and w['H11_mode']==w['H9_mode']==0 and w['H12_mode']==mode
 assert len(layouts)==16 and {z['rank'] for z in layouts}==set(range(16))
 assert {z['pid'] for z in layouts}==set(guards['167']['worker_namespace_pids'].values())
 for z in layouts:
  assert z['mode']==mode and z['transition']==t and len(z['rows'])==4 and all(v['use_cache'] for v in z['rows'])
  for key,stride in (('baseline',[128,1]),('candidate',[64,1])):
   assert len(z[key])==2 and all(a['shape']==[1048576,64] and a['stride']==stride and a['dtype']=='torch.bfloat16' and a['format']==2 for a in z[key])
  assert all(v['selected']==z['baseline' if mode==0 else 'candidate'] for v in z['rows'])
  assert all(all(a['data_ptr']==b['data_ptr'] for a,b in zip(v['output'],v['persistent'])) for v in z['rows'])
  assert all(all(a['shape']==[2,1,1,64] for a in v['output']) for v in z['rows'])
 assert len(w['H9_rows'])==16 and all('FULL' in z['runtime_mode'] and z['capture_num_graphs']>=1 and z['eager_breaks']==0 for z in w['H9_rows'])
 assert len(w['H11_rows'])==16 and all(z['mode']==0 and all(not v['actual_replay'] and v['selected_runtime']=='NONE' for v in z['rows']) for z in w['H11_rows'])
 if label!='priming_layout':assert read(label.replace('_warm','_complete')+'_witness.json')['H12_rows']==layouts
for rank in range(16):
 zs=[next(z for z in w['H12_rows'] if z['rank']==rank) for w in witnesses]
 assert all(z['baseline']==zs[0]['baseline'] and z['candidate']==zs[0]['candidate'] for z in zs)
 assert all([v['persistent'] for v in z['rows']]==[v['persistent'] for v in zs[0]['rows']] for z in zs)
retained=read('retained_witness.json');terminal_mode=retained['H12_mode']
assert retained['H6_mode']==retained['H5_event_mode']==1 and retained['H11_mode']==0
terminal_transition=initial+5+(terminal_mode!=1)
assert len(retained['H12_rows'])==16 and all(z['mode']==terminal_mode and z['transition']==terminal_transition for z in retained['H12_rows'])
assert len(retained['H11_rows'])==len(retained['H5_rows'])==16
assert all(z['returned_none'] and z['mode']==1 for z in retained['H5_rows'])
assert all(z['dispatch_cached_true'] and z['combine_cached_true'] and z['mode']==1 for z in retained['H6']['rows'])
assert all(z['health']==200 and z['idle'] and len(z['device_owners'])==16 for z in guards.values());assert all(guards[h]['root']==before[h]['root'] and guards[h]['worker_start_ticks']==before[h]['worker_start_ticks'] for h in guards)
complete={z['label']:z for z in rows if z['label'].endswith('_complete')};a1,b1,a2,b2=[complete[l+'_complete'] for l in ('A1','B1','A2','B2')]
drift_d=abs(a1['D_wall_s']-a2['D_wall_s']);drift_pd=abs(a1['PD_wall_s']-a2['PD_wall_s']);comparisons=[]
for a,b in ((a1,b1),(a2,b2)):
 comparisons.append(dict(D_saving_s=a['D_wall_s']-b['D_wall_s'],PD_saving_s=a['PD_wall_s']-b['PD_wall_s'],P_saving_s=a['P_wall_s']-b['P_wall_s'],TPOT_saving_ms=a['TPOT_ms']-b['TPOT_ms']))
same_work=all(z['workload_signature']==a1['workload_signature'] for z in (a1,b1,a2,b2));same_cache=all(z['cache']==a1['cache'] for z in (a1,b1,a2,b2))
positive=same_work and same_cache and all(z['D_saving_s']>drift_d and z['PD_saving_s']>drift_pd and z['TPOT_saving_ms']>0 for z in comparisons)
expected_verdict='POSITIVE_LIMITED_RESEARCH' if positive else 'INCONCLUSIVE';decision=read('decision.json');assert decision['verdict']==expected_verdict and decision['terminal_verified'] and not decision['recovery_used'] and decision['terminal_error'] is None
assert decision['comparisons']==comparisons and decision['D_A1_A2_drift_s']==drift_d and decision['PD_A1_A2_drift_s']==drift_pd
stack=read('retained_stack.json');assert stack['H6'] and stack['H5'] and not stack['H11'] and stack['H12']==positive and terminal_mode==int(positive)
assert decision['H12_research_stack']==positive and decision['all_complete_work_signatures_match']==same_work and decision['all_complete_cache_query_work_matches']==same_cache
assert read('functional_plan.json')['script_sha256']==hashlib.sha256((r/'pd_compare.py').read_bytes()).hexdigest()
row=dict(independent_raw_reduction=True,measurement_valid=True,work_matched=same_work,cache_matched=same_cache,verdict=expected_verdict,H12_research_stack=positive,requests=rows,unique_salts=salts,all16_transfer=True,all16_layout_witness=True,target_FULL=True,H11_mode=0,same_workers=True,H6_H5_retained=True,H12_mode=terminal_mode,terminal_transition=terminal_transition,terminal_health_idle=True,recovery_used=False,D_drift_s=drift_d,PD_drift_s=drift_pd,comparisons=comparisons,Current=None,formal_product_KEEP=False,limitations='Two complete natural23 EOS repetitions per path, strict same cache/work signatures. Aggregate acceptance counts are not ordered engine trace; SSE timings are not pure device step. Not formal80K/600/93%.')
p=r/'measurement_reduced.json';assert not p.exists();p.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ('requests','unique_salts')}))
