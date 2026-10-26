import sys,json,pathlib,hashlib,statistics,math,copy
root=pathlib.Path('/data/wio/Inference_Foundry');sys.path.insert(0,str(root/'scripts'));import extreme_bound_calibration_v3_29 as m
out=root/'evidence/20260927_loop080_bound/run548';out.mkdir(parents=True,exist_ok=True)
gen=root/'scripts/extreme_bound_calibration_v3_29.py';artifact=root/'evidence/20260927_loop080_bound/run547/bound_calibration_v3_29.json';checks=[]
def ck(n,v):
 checks.append({'name':n,'passed':bool(v)})
 if not v:raise AssertionError(n)
a=m.build();b=m.build();ck('two_memory_builds_equal',a==b);ck('artifact_byte_identical',json.dumps(a,ensure_ascii=False,indent=2)+'\n'==artifact.read_text());ck('revision',a['model_revision']=='V3.29')
ck('current_unchanged',a['current']['accepted_formal_tps']==571.681);ck('all19_proof_nodes_false',len(a['proof_dag']['certified'])==19 and not any(a['proof_dag']['certified'].values()))
for p,h in m.HASHES.items():
 ck('direct_sha_'+p,hashlib.sha256((root/p).read_bytes()).hexdigest()==h)
 old=m.HASHES[p];m.HASHES[p]='0'*64
 try:m.build();ok=False
 except ValueError:ok=True
 finally:m.HASHES[p]=old
 ck('pin_corruption_rejected_'+p,ok)
rows=[];r=root/'evidence/20260927_loop079_identity/run542';paths=[]
for p in (r/'ledger').glob('pid*.jsonl'):
 if '.flush.' not in p.name:paths.append(p);rows += [json.loads(l) for l in p.read_text().splitlines()]
rr=[x for x in rows if x.get('phase')=='measured'];bulk={x['request_id']:x for x in rr if x['event']=='scheduler_append' and x['bulk']};drains=[x for x in rr if x['event']=='runtime_post_drain' and x['rank']==0];hands={rid:x for x in rr if x['event']=='runtime_handoff' and x['rank']==0 for rid in x['req_ids']};ordinary=[x for x in rr if x['event']=='scheduler_append' and not x['bulk']]
gh=sum(len(x['admitted_raw_ids']) for x in ordinary if x['after_ns']<=hands[x['request_id']]['monotonic_ns']);gb=sum(x['g_before'] for x in bulk.values());q=sum(sum(sum(c) for c in d['count_history']) for d in drains)
new=a['bound_semantics_v3_29'];account=new['measured_output_accounting'];ck('G_H_independent_raw',gh==account['ordinary_G_at_rank0_handoff']==400);ck('G_prebulk_independent_raw',gb==account['scheduler_ordinary_before_terminal_bulk_G']==401);ck('q_independent_raw',q==account['runtime_sampled_q_before_remaining_clip']==49462);ck('bulk_independent_raw',sum(len(x['admitted_raw_ids']) for x in bulk.values())==account['runtime_terminal_bulk_admitted']==48751);ck('R_independent_raw',sum(sum(len(z) for z in d['retained_raw_ids']) for d in drains)==account['runtime_retained_R']==49152)
ck('first_runtime_token_remains',all(x['incoming_raw_ids'][0]==x['admitted_raw_ids'][0] for x in bulk.values()))
durations=[];chain=0;ideal_chain=0
for d in drains:
 ds=[]
 for slot,rid in enumerate(d['req_ids']):
  target=1024-bulk[rid]['g_before'];total=0
  for i,c in enumerate(d['count_history']):
   total+=c[slot]
   if total>=target:ds.append(i+1);break
 durations+=ds;chain+=max(ds);ideal_chain+=max(math.ceil((1024-bulk[rid]['g_before'])/8) for rid in d['req_ids'])
cycle={'remaining_bulk_slot_load_ceil':math.ceil((49152-gb)/96),'per_request_ideal8_acceptance_slot_load_ceil':math.ceil(sum(math.ceil((1024-x['g_before'])/8) for x in bulk.values())/12),'four_fixed_cohort_ideal8_chain_sum':ideal_chain,'observed_q_duration_clipped_bulk_arbitrary_slot_load_ceil':math.ceil(sum(durations)/12),'observed_q_duration_clipped_bulk_fixed_cohort_chain_sum':chain,'executed_cycle_sum':sum(d['cycles'] for d in drains)}
ck('all_six_cycle_counts_independent_raw',all(new['conditional_same_trajectory_cycle_relaxations'][k]==v for k,v in cycle.items()))
# Explicit independent null check; unlike a name-based validator this includes all current new fields.
v28='bound_semantics_v3_28.';v29='bound_semantics_v3_29.proof_impact.'
nullpaths=[v28+'architecture_class.unrestricted_semantic_equivalence.positive_measured_window_Target_W_minus',v28+'architecture_class.declared_online_inference.fresh_required_F_layer_group',v28+'resource_relaxation.attainable_C_plus_B',v28+'resource_relaxation.finite_architecture_floor_s',v28+'scheduling_execution.necessary_path_floor_s',v28+'scheduling_execution.feasible_resource_contended_schedule_s',v28+'product_e2e.strict_tps_ceiling',v28+'product_e2e.conditional_predictive_interval_tps',v28+'product_e2e.distance_from_current_to_credible_limit_tps','bound_ladder.scheduling_execution.run537_architecture_variant_review.all8_mixed_service_frontier','bound_ladder.algorithm_resource.run538_semantic_witness_limit.formal_F_layer_group']+[v29+k for k in ['algorithm_resource_latency_floor_s','hardware_resource_latency_floor_s','scheduling_execution_latency_floor_s','product_e2e_tps_ceiling','distance_from_formal_current_571_681_to_limit_tps']]+['bound_ladder.scheduling_execution.run542_546_host_lineage.'+k for k in ['cached_residual_device_ready','all8_mixed_service_frontier','feasible_alternative_schedule']]
def val(x,path):
 for k in path.split('.'):x=x[k]
 return x
ck('explicit19_extended_unproved_fields_null',all(val(a,p) is None for p in nullpaths))
m.require_null_endpoints(a)
orig=m.require_null_endpoints;neg=[]
for path in nullpaths:
 def inject(tree,path=path):
  orig(tree);node=tree;ks=path.split('.')
  for k in ks[:-1]:node=node[k]
  node[ks[-1]]=1.0
 m.require_null_endpoints=inject
 try:m.build();rejected=False
 except ValueError:rejected=True
 finally:m.require_null_endpoints=orig
 neg.append({'path':path,'rejected':rejected})
# Truth of repr/clock provenance remains explicitly separate from device evidence.
ck('prefill_repr_scope','repr only' in new['measured_prefill_stats_repr']['scope']);ck('cycle_relaxation_scope','not Product time lower bounds' in new['conditional_same_trajectory_cycle_relaxations']['scope']);ck('Host_span_scope','No sum or subtraction' in new['instrumented_host_spans']['scope']);ck('fresh_F_not_certified',a['bound_ladder']['algorithm_resource']['run542_546_same_run_output_lineage']['fresh_required_Target_F_certified'] is False)
manifest={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(set(a['input_paths'])|{str(p.relative_to(root)) for p in paths}|{'scripts/extreme_bound_calibration_v3_29.py','evidence/20260927_loop080_bound/run547/bound_calibration_v3_29.json'})}
result={'status':'PASS' if all(x['rejected'] for x in neg) else 'CURRENT_ARTIFACT_PASS_GENERATOR_GUARD_FAIL','checks':checks,'full_generator_nonnull_injection_controls':neg,'raw_recomputed_cycles':cycle,'input_sha256':manifest,'generator_sha256':hashlib.sha256(gen.read_bytes()).hexdigest(),'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'service_or_NPU_execution':False}
(out/'independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','generator_sha256','artifact_sha256','raw_recomputed_cycles']},indent=2));print('checks',len(checks),'injection_pass',sum(x['rejected'] for x in neg),'/',len(neg),'hashes',len(manifest))
