import json,pathlib,hashlib,re,sqlite3,struct,ast,statistics,datetime,sys,math
d=pathlib.Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/REDUCE-FULL43-20261002T0057Z");root=d.parents[4];sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'runtime'))
from protocol_receipts import verify_protocol_receipts
from phase_runner import same_process
r=d.parents[1]/'runs/GLM-RUN-0043'
state=json.loads((r/'state.json').read_text());assert state['status']=='completed' and not same_process(state['owner'])
spec=json.loads((r/'controller_spec.json').read_text())
for pin in spec['stages'][0]['sources']:
 assert hashlib.sha256(pathlib.Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
 assert pathlib.Path(pin['path']).read_bytes()==pathlib.Path(pin['snapshot']).read_bytes()
f=r/'formal'
base=json.loads((r/'reduction.json').read_text());assert base['measurement_valid']
receipt=verify_protocol_receipts(f/'router_trace.jsonl',4,81932,61440,{'TP32'});assert receipt['effective_output_tokens']==245760
trace=[json.loads(l) for l in (f/'router_trace.jsonl').read_text().splitlines()]
by={}
for e in trace:
 if e.get('lease_id'):by.setdefault(e['lease_id'],{})[e['event']]=e
rows=[]
for x in base['requests']:
 events=by[x['lease_id']];a=events['lease_acquired'];o=events['upstream_first_output'];end=events['upstream_stream_contract']
 raw=pathlib.Path(x['wire']['path']).read_bytes();assert len(raw)==x['wire']['bytes'] and hashlib.sha256(raw).hexdigest()==x['wire']['sha256']
 rows.append({'lease_id':x['lease_id'],'replica':x['replica'],'body_sha256':x['body_sha256'],'wire':x['wire'],'gateway_first_output_s':(o['monotonic_ns']-a['monotonic_ns'])/1e9,'gateway_request_s':(end['monotonic_ns']-a['monotonic_ns'])/1e9,'native_usage':x['usage'],'strict_valid':x['valid']})
def metrics(p):
 out={}
 for line in p.read_text().splitlines():
  m=re.match(r'(vllm:[^ {]+)(?:\{([^}]+)\})?\s+([0-9.eE+\-]+)$',line)
  if m:out[(m[1],m[2] or '')]=float(m[3])
 return out
def total(m,name):
 return sum(v for (k,labels),v in m.items() if k=='vllm:'+name)
native={}
for replica in ['TP32']:
 events=json.loads((f/'formal_events.json').read_text());full_start=datetime.datetime.fromisoformat(json.loads((f/'benchmark/full_execution.json').read_text())['started_at']);warm_end=datetime.datetime.fromisoformat(json.loads((f/'benchmark/warmup_execution.json').read_text())['finished_at'])
 idle=[e for e in events if e['event']=='metrics' and warm_end<=datetime.datetime.fromisoformat(e['at'])<full_start and all(v==0 for x in e['gauges'].values()for v in x.values())]
 assert idle,"No verified full-only idle counter baseline";label=idle[-1]['label']
 p=f/(label+'_'+replica+'.metrics');z=f/('final_'+replica+'.metrics');a=metrics(p);b=metrics(z)
 names=['prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_queries_total','external_prefix_cache_hits_total','num_preemptions_total','generation_tokens_total','spec_decode_num_drafts_total','spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total','request_prefill_time_seconds_sum','request_prefill_time_seconds_count','request_queue_time_seconds_sum','time_to_first_token_seconds_sum','time_to_first_token_seconds_count']
 delta={n:total(b,n)-total(a,n) for n in names};assert delta['generation_tokens_total']==245760
 drafts=delta['spec_decode_num_drafts_total'];accepted=delta['spec_decode_num_accepted_tokens_total']
 pos=[]
 for i in range(5):
  def v(m):return sum(v for (k,l),v in m.items() if k=='vllm:spec_decode_num_accepted_tokens_per_pos_total' and 'position="'+str(i)+'"' in l)
  pos.append((v(b)-v(a))/drafts)
 native[replica]={'window':label+' verified idle after one warmup to final idle; all4 full requests on one scheduler','before':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'after':{'path':str(z),'sha256':hashlib.sha256(z.read_bytes()).hexdigest()},'delta':delta,'accepted_per_draft_sequence':accepted/drafts,'expected_work_per_draft_sequence':(drafts+accepted)/drafts,'accepted_per_position_per_draft_sequence':pos,'accepted_plus_bonus_minus_committed':drafts+accepted-245760,'cache_uncached_query_tokens':delta['prefix_cache_queries_total']-delta['prefix_cache_hits_total']}
details=pathlib.Path(base['slo']['details_jsonl']);ais=[]
dbs=list((details.parent/'db_data').glob('*.db'));assert len(dbs)==1;db=dbs[0];c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
for line in details.read_text().splitlines():
 x=json.loads(line);blob=c.execute('select arr_blob from numpy_store where id=?',(x['time_points']['__db_ref__'],)).fetchone()[0]
 assert blob[:8]==b'\x93NUMPY\x01\x00';n=struct.unpack('<H',blob[8:10])[0];h=ast.literal_eval(blob[10:10+n].decode());assert h['descr']=='<f8' and not h['fortran_order'] and len(h['shape'])==1
 points=struct.unpack('<'+'d'*h['shape'][0],blob[10+n:]);assert all(math.isfinite(t) for t in points)
 ais.append({'data_id':x['data_id'],'uuid':x['uuid'],'native_output_tokens':x['output_tokens'],'native_input_tokens':x['input_tokens'],'stored_time_point_count':len(points),'ttft_first_sse_s':points[1]-points[0],'e2el_s':points[-1]-points[0],'tpot_s':(points[-1]-points[1])/(x['output_tokens']-1),'time_array_blob_sha256':hashlib.sha256(blob).hexdigest()})
out={'run_id':'GLM-RUN-0043','measurement_valid':True,'formal_summary':base,'gateway_requests':rows,'native_full_only':native,'ais_unrounded':ais,'raw_tpot_p50_ms':statistics.median(x['tpot_s'] for x in ais)*1000,'raw_ttft_p50_ms':statistics.median(x['ttft_first_sse_s'] for x in ais)*1000,'slo_verdict_original_unchanged':base['slo']['slo'],'limits':['Native histograms are host-engine timestamps, not device-kernel attribution','Draft counter counts issued sequences, not batched physical engine iterations','Native accepted-plus-bonus overshoot is retained, not effective output credit','Time-array packet count is not output token count','One valid full round does not establish repeated KEEP or stable online capacity','Raw unrounded diagnostic does not change original CSV-based SLO policy']}
warm=verify_protocol_receipts(f/'router_trace.jsonl',1,73740,1,{'TP32'});assert warm['effective_output_tokens']==1
samples=[]
for path in f.glob('*_TP32.metrics'):
 m=metrics(path)
 samples.append({'path':str(path),'kv_cache_usage':total(m,'kv_cache_usage_perc'),'running':total(m,'num_requests_running'),'waiting':total(m,'num_requests_waiting'),'preemptions':total(m,'num_preemptions_total')})
out['native_highwater']={'max_kv_cache_usage':max(x['kv_cache_usage']for x in samples),'max_running':max(x['running']for x in samples),'max_waiting':max(x['waiting']for x in samples),'samples':len(samples),'highest_samples':sorted(samples,key=lambda x:x['kv_cache_usage'],reverse=True)[:5]}
assert total(metrics(f/'final_TP32.metrics'),'num_requests_running')==total(metrics(f/'final_TP32.metrics'),'num_requests_waiting')==0
out['source_pins']=len(spec['stages'][0]['sources']);out['warmup_receipt']=warm
out['native_models']=json.loads((r/'adopted_model_identities.json').read_text())
out['cache_condition']=json.loads((f/'cache_condition.json').read_text())
out['limits'].extend(['Resident native Run42 cache retained: first two full prompts previously probed; cache equality versus Run21 not established','Finite C2 window and sampled KV highwater do not certify worst-case simultaneous terminal contexts or stable capacity'])
p=d/'reduction.json';p.write_text(json.dumps(out,indent=2))
raw=p.read_bytes();result={'schema_version':1,'job_id':d.name,'status':'completed','summary':'Run43 completed strict full E2E; four raw hashes, full-only native cache/decode counters and unrounded AIS times reduced without inference','execution':{'inner_exit_code':0,'acceptance':'passed','processes':[]},'findings':[{'kind':'fact','text':'4 full requests native output245760 verified; original SLO policy preserved','scope':{'run':'GLM-RUN-0043','full_only_counter_window':'verified idle baseline->final','effective_tps':base['effective_tps_full_cli_phase']},'evidence_ids':['reduction']}],'evidence':[{'id':'reduction','path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'locator':'native_full_only and raw AIS array diagnostics'}],'unknowns':out['limits'],'decision_request':None,'next_check_at':None}
(d/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'native':native,'raw_tpot_p50_ms':out['raw_tpot_p50_ms'],'raw_ttft_p50_ms':out['raw_ttft_p50_ms']}))
