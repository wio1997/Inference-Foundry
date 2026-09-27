import pathlib,json,collections,statistics,math,hashlib,re
root=pathlib.Path('/data/wio/Inference_Foundry');r=root/'evidence/20260927_loop079_identity/run542';out=root/'evidence/20260927_loop080_bound/run545';out.mkdir(parents=True,exist_ok=True)
paths=[]
def read(p):
 paths.append(p);return json.loads(p.read_text())
rows=[]
for p in sorted((r/'ledger').glob('pid*.jsonl')):
 if '.flush.' in p.name:continue
 paths.append(p);rows += [json.loads(l) for l in p.read_text().splitlines()]
def stat(a):return {'min':min(a),'median':statistics.median(a),'max':max(a),'sum':sum(a),'n':len(a)}
result={'scope':'conditional_on_Run544_admission; instrumented_Host_accounting_only','phases':{},'cross_process_durations_computed':False}
request_outputs={};request_prompts={}
for phase in ['warmup','measured']:
 rr=[x for x in rows if x.get('phase')==phase];events=collections.defaultdict(list)
 for x in rr:events[x['event']].append(x)
 client=read(r/(phase+'_client')/'summary.json'); cr=client['requests'];summary=client['summary']
 add={x['request_id']:x for x in events['output_add_request']};external={k:v['external_req_id'] for k,v in add.items()};byext={x['response_id']:x for x in cr}
 bulks={x['request_id']:x for x in events['scheduler_append'] if x['bulk']};assert len(bulks)==48
 ordinary=collections.defaultdict(list)
 for x in sorted(events['scheduler_append'],key=lambda x:x['seq']):
  if not x['bulk']:ordinary[x['request_id']]+=x['admitted_raw_ids']
 G=[x['g_before'] for x in bulks.values()]; assert all(len(ordinary[k])==v['g_before'] for k,v in bulks.items())
 request_outputs[phase]={byext[external[k]]['i']:ordinary[k]+v['admitted_raw_ids'] for k,v in bulks.items()}
 request_prompts[phase]={byext[external[k]]['i']:add[k]['prompt_sha256'] for k in bulks}
 drains=sorted([x for x in events['runtime_post_drain'] if x['rank']==0],key=lambda x:x['cohort']);cohorts=[];slot_rows=[]
 for d in drains:
  c=d['cohort'];q=d['count_history'];assert len(q)==d['cycles'];assert all(0<=v<=8 for z in q for v in z)
  peer=[x for x in events['runtime_post_drain'] if x['cohort']==c];assert sorted(x['rank'] for x in peer)==list(range(8))
  for x in peer:assert all(x[k]==d[k] for k in ['req_ids','count_history','retained_raw_ids','staged_counts','initial_positions','token_history'])
  g=[];dc=[];er=[];sumq=0
  for s,rid in enumerate(d['req_ids']):
   b=bulks[rid];g.append(b['g_before']);staged=[];S=0;finish1024=None;finish_ext=None
   for cyc,counts in enumerate(q):
    count=counts[s];staged+=d['token_history'][cyc][s][:count];S+=count
    if S>=1024 and finish1024 is None:finish1024=cyc+1
    if S>=1024-b['g_before'] and finish_ext is None:finish_ext=cyc+1
   retained=staged[:1024];assert retained==d['retained_raw_ids'][s]==b['incoming_raw_ids'];assert b['admitted_raw_ids']==retained[:1024-b['g_before']]
   assert b['g_after']==1024 and b['g_after']-b['g_before']==len(b['admitted_raw_ids'])
   assert retained[0]==b['admitted_raw_ids'][0]
   dc.append(finish1024);er.append(finish_ext);sumq+=S
   slot_rows.append({'request_id':rid,'external_id':external[rid],'dataset_index':byext[external[rid]]['i'],'cohort':c,'slot':s,'G':b['g_before'],'runtime_q_sum':S,'runtime_retained':1024,'scheduler_bulk_admitted':1024-b['g_before'],'cycle_count_to_1024':finish1024,'cycle_count_to_1024_minus_G':finish_ext,'external_first_runtime_offset':b['g_before']})
  rankspans=[];h2first=[];last2drain=[];drain2done=[]
  for rank in range(8):
   h=next(x for x in events['runtime_handoff'] if x['rank']==rank and x['cohort']==c);pd=next(x for x in peer if x['rank']==rank);done=next(x for x in events['runner_done'] if x['rank']==rank and x['cohort']==c);cy=sorted([x for x in events['serving_host_cycle'] if x['rank']==rank and x['cohort']==c],key=lambda x:x['cycle'])
   assert h['pid']==pd['pid']==done['pid']==cy[0]['pid'];assert len(cy)==d['cycles']
   rankspans.append((pd['monotonic_ns']-h['monotonic_ns'])/1e9);h2first.append((cy[0]['monotonic_ns']-h['monotonic_ns'])/1e6);last2drain.append((pd['monotonic_ns']-cy[-1]['monotonic_ns'])/1e6);drain2done.append((done['monotonic_ns']-pd['monotonic_ns'])/1e6)
  clients=[byext[external[k]] for k in d['req_ids']]
  cohorts.append({'cohort':c,'cycles':d['cycles'],'G_sum':sum(g),'G':g,'q_sum':sumq,'runtime_retained':12288,'scheduler_bulk_admitted':12288-sum(g),'q_excess_over_runtime_retained':sumq-12288,'duration_cycles_to_1024':dc,'duration_cycles_to_1024_minus_G':er,'rank_handoff_to_drain_s':stat(rankspans),'rank_handoff_to_first_host_cycle_ms':stat(h2first),'rank_last_host_cycle_to_drain_ms':stat(last2drain),'rank_drain_to_done_ms':stat(drain2done),'client_submit_spread_ms':(max(x['start_monotonic_ns'] for x in clients)-min(x['start_monotonic_ns'] for x in clients))/1e6,'client_completion_spread_ms':(max(x['end_monotonic_ns'] for x in clients)-min(x['end_monotonic_ns'] for x in clients))/1e6,'client_first_submit_to_last_end_s':(max(x['end_monotonic_ns'] for x in clients)-min(x['start_monotonic_ns'] for x in clients))/1e9})
 # All following Host intervals are within one PID, never subtract client/server clocks.
 api_done={x['external_req_id']:x for x in events['api_serialized_yield'] if x['payload'].strip()=='data: [DONE]'}
 api_first={}
 for x in events['output_receive']:api_first.setdefault(x['request_id'],x)
 api_add_first=[];api_first_done=[];prefill=[]
 for rid,a in add.items():
  f=api_first[rid];done=api_done[external[rid]];assert a['pid']==f['pid']==done['pid'];api_add_first.append((f['monotonic_ns']-a['monotonic_ns'])/1e6);api_first_done.append((done['monotonic_ns']-f['monotonic_ns'])/1e6)
  for x in events['output_receive']:
   if x['request_id']==rid and x['prefill_stats'] not in ['None',None]:
    nums={k:int(v) for k,v in re.findall(r'(num_\w+)=(\d+)',x['prefill_stats'])};prefill.append({'dataset_index':byext[external[rid]]['i'],**nums});break
 # Rank-ordered completions -> FIFO queued admission is a client-only chronology relation, not proof of scheduler readiness.
 end=sorted(cr,key=lambda x:x['end_monotonic_ns'])[:36];start=sorted(cr,key=lambda x:x['start_monotonic_ns'])[12:];gaps=[(s['start_monotonic_ns']-e['end_monotonic_ns'])/1e6 for s,e in zip(start,end)]
 inter=[]
 for a,b in zip(drains,drains[1:]):
  done=next(x for x in events['runner_done'] if x['rank']==0 and x['cohort']==a['cohort']);h=next(x for x in events['runtime_handoff'] if x['rank']==0 and x['cohort']==b['cohort']);assert done['pid']==h['pid'];inter.append((h['monotonic_ns']-done['monotonic_ns'])/1e3)
 dur=[s['cycle_count_to_1024_minus_G'] for s in slot_rows];rt=[s['cycle_count_to_1024'] for s in slot_rows]
 result['phases'][phase]={'G':stat(G),'cycles_total':sum(c['cycles'] for c in cohorts),'q_total':sum(c['q_sum'] for c in cohorts),'runtime_retained_total':49152,'ordinary_total':sum(G),'scheduler_bulk_total':49152-sum(G),'scheduler_total':49152,'q_excess':sum(c['q_sum'] for c in cohorts)-49152,'all48_first_runtime_token_retained':True,'cohorts':cohorts,'per_request':slot_rows,'client_summary':summary,'client_rank_matched_completion_to_next_submit_ms':stat(gaps),'client_rank_matched_completion_to_next_submit_all_nonnegative':all(g>=0 for g in gaps),'api_add_to_first_output_receive_ms':stat(api_add_first),'api_first_output_receive_to_DONE_ms':stat(api_first_done),'rank0_done_to_next_handoff_us':inter,'prefill_stats_source_repr':prefill,'conditional_count_relaxations':{'external_bulk_cardinality_slot_load':math.ceil((49152-sum(G))/96),'per_request_atmost8_slot_load':math.ceil(sum(math.ceil((1024-g)/8) for g in G)/12),'fixed_cohort_atmost8_serial':sum(max(math.ceil((1024-g)/8) for g in c['G']) for c in cohorts),'observed_duration_slot_load_external_clipped':math.ceil(sum(dur)/12),'observed_duration_fixed_cohort_serial_external_clipped':sum(max(c['duration_cycles_to_1024_minus_G']) for c in cohorts),'observed_duration_slot_load_runtime1024':math.ceil(sum(rt)/12),'observed_duration_fixed_cohort_serial_runtime1024':sum(max(c['duration_cycles_to_1024']) for c in cohorts),'scope':'cycle-domain relaxations for this observed q trajectory only; no preparation/release/device-time/feasible-overlap proof'}}
# Compare exact scheduler raw sequences by frozen dataset ordinal; no semantic-equivalence or residency inference.
match=[];prefix=[]
for i in range(48):
 a=request_outputs['warmup'][i];b=request_outputs['measured'][i];n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 prefix.append(n)
 if a==b:match.append(i)
assert request_prompts['warmup']==request_prompts['measured']
result['warmup_measured_comparison']={'prompt_hash_match_count':48,'scheduler_raw_output_exact_match_count':len(match),'common_raw_prefix_lengths':prefix,'common_raw_prefix_stats':stat(prefix),'scope':'observed raw sequence equality only, no bitwise hidden-state equivalence or reusable residency claim'}
paths += [r/'cleanup_status.txt',r/'server_validate.log',root/'evidence/20260927_loop079_identity/run543/server_admission_replayed.json']
result['input_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
(out/'same_trajectory_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for phase,v in result['phases'].items():
 print(phase,{k:x for k,x in v.items() if k not in ['cohorts','per_request','client_summary','prefill_stats_source_repr']})
 print('cohort summary',[(c['cohort'],c['cycles'],c['G_sum'],c['rank_handoff_to_drain_s']['median']) for c in v['cohorts']]);print('prefill',collections.Counter((x.get('num_prompt_tokens'),x.get('num_computed_tokens'),x.get('num_cached_tokens')) for x in v['prefill_stats_source_repr']))
print(result['warmup_measured_comparison'])
