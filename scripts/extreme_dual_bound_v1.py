#!/usr/bin/env python3
"""Executable dependency skeleton for the frozen Extreme cycle.

Only supplied, same-path node costs produce a numeric critical-path relaxation.
It never promotes standalone kernel time, trace bytes, or V0 assumptions to a
hardware ceiling. Resource-constrained scheduling and Product E2E remain open.
"""
import argparse, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NODES={
 'target_k': {'resource':['AIC','AIV','HBM','HCCL'], 'meaning':'target verified hidden/logits and target KV writes'},
 'acceptance_k': {'resource':['AIV','HBM'], 'meaning':'accepted counts, last sampled token, selection'},
 'state_advance_k': {'resource':['AIV','HBM'], 'meaning':'advance computed tokens and fixed state'},
 'dspark_old_state_k': {'resource':['AIV','HBM','CPU'], 'meaning':'refresh common, context slots, gather old target hidden/ids/positions'},
 'dspark_model_k': {'resource':['AIC','AIV','HBM','HCCL','CPU'], 'meaning':'draft7 model and next draft output'},
 'metadata_k1': {'resource':['AIV','AICPU','HBM','CPU'], 'meaning':'next positions/lengths/slots/RoPE/SAS/QLI'},
 'metadata_commit_k1': {'resource':['AIV','HBM'], 'meaning':'copy next scratch to fixed Graph addresses; zero in current serial layout'},
 'next_ids_k1': {'resource':['AIV','HBM'], 'meaning':'next target ids from accepted token and DSpark draft'},
 'target_k1': {'resource':['AIC','AIV','HBM','HCCL'], 'meaning':'following target verification'},
}
COMMON=[('target_k','acceptance_k','semantic_RAW'),('target_k','dspark_old_state_k','semantic_RAW'),('acceptance_k','state_advance_k','semantic_RAW'),('state_advance_k','dspark_old_state_k','semantic_RAW'),('dspark_old_state_k','dspark_model_k','semantic_RAW'),('dspark_model_k','next_ids_k1','semantic_RAW'),('next_ids_k1','target_k1','semantic_RAW'),('target_k','target_k1','KV_state_RAW')]
CURRENT=COMMON+[('dspark_model_k','metadata_k1','implementation_order'),('next_ids_k1','metadata_k1','implementation_order'),('state_advance_k','metadata_k1','semantic_RAW'),('metadata_k1','target_k1','semantic_RAW')]
OVERLAP=COMMON+[('state_advance_k','metadata_k1','semantic_RAW'),('metadata_k1','metadata_commit_k1','semantic_RAW'),('dspark_model_k','metadata_commit_k1','storage_WAR_guard'),('metadata_commit_k1','next_ids_k1','implementation_order'),('metadata_commit_k1','target_k1','semantic_RAW')]

# Nested frozen c4 Target DSA CP path. The current product executes the
# context_parallel/dsa_cp.py override, not the dsa_v1.py CV multistream path.
CP_C4_NODES={
 'h_local': {'resource':[], 'meaning':'already available local hidden input'},
 'h_full': {'resource':['HCCL','HBM'], 'meaning':'gathered full hidden for KV/cache updates'},
 'q_local': {'resource':['AIC','AIV','HBM'], 'meaning':'local Q projection, RMS, RoPE'},
 'swa_kv': {'resource':['AIC','AIV','HBM'], 'meaning':'SWA KV projection, RoPE, cache scatter'},
 'indexer_cache': {'resource':['AIC','AIV','HBM'], 'meaning':'indexer compressor, rotate/quant and key/scale scatter'},
 'indexer_qli': {'resource':['AIC','AIV','HBM'], 'meaning':'indexer query/weights and QLI topk'},
 'main_compressor': {'resource':['AIC','AIV','HBM'], 'meaning':'main compressor and compressed KV scatter'},
 'sparse_attention': {'resource':['AIC','AIV','HBM'], 'meaning':'join of Q, SWA KV, topk and compressed KV'},
 'output': {'resource':['AIC','AIV','HBM','HCCL'], 'meaning':'restore head layout and output projection'},
}
CP_C4_SEMANTIC=[
 ('h_local','h_full','semantic_RAW'),('h_local','q_local','semantic_RAW'),
 ('h_full','swa_kv','semantic_RAW'),('h_full','indexer_cache','semantic_RAW'),
 ('indexer_cache','indexer_qli','semantic_RAW'),('q_local','indexer_qli','semantic_RAW'),
 ('h_full','main_compressor','semantic_RAW'),
 ('q_local','sparse_attention','semantic_RAW'),('swa_kv','sparse_attention','semantic_RAW'),
 ('indexer_qli','sparse_attention','semantic_RAW'),
 ('main_compressor','sparse_attention','semantic_RAW'),
 ('sparse_attention','output','semantic_RAW'),
]
CP_C4_CURRENT=CP_C4_SEMANTIC+[
 ('h_full','q_local','implementation_order'),('q_local','swa_kv','implementation_order'),
 ('swa_kv','indexer_cache','implementation_order'),
 ('indexer_qli','main_compressor','implementation_order'),
]
CP_C4_FORK=CP_C4_SEMANTIC+[
 ('h_full','q_local','implementation_order'),('q_local','swa_kv','implementation_order'),
 ('swa_kv','indexer_cache','implementation_order'),
 ('indexer_cache','main_compressor','fork_after_indexer_cache'),
]
# H003: hidden AllGather and independent local Q can issue from h_local;
# the gathered hidden still joins before WKV/cache consumers. Run292 proves
# one-layer device overlap, not a whole-cycle or Product speedup.
CP_C4_HIDDEN_GATHER=CP_C4_SEMANTIC+[
 ('q_local','swa_kv','implementation_order'),
 ('swa_kv','indexer_cache','implementation_order'),
 ('indexer_qli','main_compressor','implementation_order'),
]

# Unrestricted semantic relaxation: qr becomes available before main Q finishes;
# indexer query preparation can run before the indexer cache update completes.
# This is a dependency graph, not a claim that concurrent 910B3 resources exist.
CP_C4_FINE_NODES={
 'h_local': {'resource':[], 'meaning':'local hidden input'},
 'h_full': {'resource':['HCCL','HBM'], 'meaning':'gathered hidden'},
 'qr_local': {'resource':['AIC','AIV','HBM'], 'meaning':'q_a and normalization yielding qr'},
 'q_complete': {'resource':['AIC','AIV','HBM'], 'meaning':'main q_b, RMS and RoPE after qr'},
 'swa_kv': {'resource':['AIC','AIV','HBM'], 'meaning':'SWA KV projection and cache update'},
 'indexer_cache': {'resource':['AIC','AIV','HBM'], 'meaning':'indexer cache update'},
 'indexer_query': {'resource':['AIC','AIV','HBM'], 'meaning':'indexer query and weights from local hidden and qr'},
 'qli': {'resource':['AIC','AIV','HBM'], 'meaning':'indexer selection reads updated cache and prepared query'},
 'main_compressor': {'resource':['AIC','AIV','HBM'], 'meaning':'main compressor and compressed KV scatter'},
 'sparse_attention': {'resource':['AIC','AIV','HBM'], 'meaning':'join of Q, SWA KV, topk and compressed KV'},
 'output': {'resource':['AIC','AIV','HBM','HCCL'], 'meaning':'output projection and collective'},
}
CP_C4_FINE_SEMANTIC=[
 ('h_local','h_full','semantic_RAW'),('h_local','qr_local','semantic_RAW'),
 ('qr_local','q_complete','semantic_RAW'),
 ('h_full','swa_kv','semantic_RAW'),('h_full','indexer_cache','semantic_RAW'),
 ('h_full','main_compressor','semantic_RAW'),
 ('h_local','indexer_query','semantic_RAW'),('qr_local','indexer_query','semantic_RAW'),
 ('indexer_cache','qli','semantic_RAW'),('indexer_query','qli','semantic_RAW'),
 ('q_complete','sparse_attention','semantic_RAW'),('swa_kv','sparse_attention','semantic_RAW'),
 ('qli','sparse_attention','semantic_RAW'),('main_compressor','sparse_attention','semantic_RAW'),
 ('sparse_attention','output','semantic_RAW'),
]

def analyse(edges,costs,nodes=NODES,terminal='target_k1'):
 active={node for src,dst,_ in edges for node in (src,dst)}
 incoming={k:[] for k in active}
 for src,dst,kind in edges:
  assert src in nodes and dst in nodes and kind in {'semantic_RAW','KV_state_RAW','storage_WAR_guard','implementation_order','fork_after_indexer_cache'}
  incoming[dst].append(src)
 remaining=set(active);order=[]
 while remaining:
  ready=sorted(k for k in remaining if all(p in order for p in incoming[k]))
  if not ready:raise ValueError('dependency cycle')
  order.extend(ready);remaining.difference_update(ready)
 unknown=[k for k in order if costs.get(k) is None]
 if unknown:return {'topological_order':order,'critical_path_ms':None,'missing_node_costs':unknown}
 finish={}
 for k in order:finish[k]=float(costs[k])+max((finish[p] for p in incoming[k]),default=0.0)
 return {'topological_order':order,'critical_path_ms':finish[terminal],'missing_node_costs':[]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--durations-json');ap.add_argument('--cp-durations-json');a=ap.parse_args()
 # Stage medians are descriptive diagnostics; they are not same-cycle node costs
 # and cannot be summed into a complete DAG without paired timing and overlap.
 run98_stage=json.loads((ROOT/'evidence/20260924_loop036_metadata/run98/summary.json').read_text())['stage_median_ms']
 observed_stage={'target_k':('runtime','target'),'acceptance_k':('runtime','acceptance'),
                 'state_advance_k':('runtime','state_advance'),
                 'dspark_model_k':('dspark','model'),'metadata_k1':('runtime','derived_target_metadata'),
                 'next_ids_k1':('runtime','draft_commit')}
 node_cost_ledger={name:{'observed_diagnostic_current_ms':
                      run98_stage[group][key] if name in observed_stage else None,
                      'engineering_bound_ms':None,'aggressive_bound_ms':None,
                      'evidence':'Run98 eight-rank 300-cycle stage-event median' if name in observed_stage else 'not yet isolated on same path',
                      'confidence':'medium for Run98 stage median, low for Run99 same-cycle transfer' if name in observed_stage else 'unknown',
                      'scope':'marginal stage medians, not additive same-cycle DAG costs'}
                   for name in NODES
                   for group,key in [observed_stage.get(name,(None,None))]}
 costs=json.loads(Path(a.durations_json).read_text()) if a.durations_json else {}
 cp_costs=json.loads(Path(a.cp_durations_json).read_text()) if a.cp_durations_json else {}
 inventory=json.loads((ROOT/'evidence/20260926_loop060_resource/run242/inventory.json').read_text())
 compressor=json.loads((ROOT/'evidence/20260926_loop062_nongmm/run256/census.json').read_text())
 counters=json.loads((ROOT/'evidence/20260926_loop060_resource/run247/analysis.json').read_text())
 hccl=json.loads((ROOT/'evidence/20260926_loop061_bound/run250/payload.json').read_text())
 ownership=json.loads((ROOT/'evidence/20260926_loop064_cp/astra_cp_request_ownership_screen.json').read_text())
 owner287=json.loads((ROOT/'evidence/20260926_loop064_cp/run287/analysis.json').read_text())
 handoff=json.loads((ROOT/'evidence/20260926_loop064_cp/run283/analysis.json').read_text())
 parked=json.loads((ROOT/'evidence/20260926_loop065_gather/astra_existing_acceptance_screen.json').read_text())
 hidden_matched=json.loads((ROOT/'evidence/20260926_loop065_gather/run293/astra_matched_metrics.json').read_text())
 assert parked['aggregate']['inactive_slot_cycles']==2989 and parked['aggregate']['cycles']==1496
 assert hidden_matched['summary']['pre_to_wkv_us']['samples']==16
 assert len(handoff['ready_cohorts'])==5 and handoff['rank_cohort_pass']==40
 assert ownership['status']=='conditional_static_shape_screen' and len(ownership['rank_rows'])==8
 assert ownership['conditional_owned_request_rank_memberships']==16
 for row in ownership['rank_rows']:
  rank=row['rank']; start=12*rank; end=start+12
  expected=[max(0,min(end,8*(req+1))-max(start,8*req)) for req in range(12)]
  assert row['query_flat_interval']==[start,end]
  assert row['local_query_lengths_by_request']==expected
  assert row['request_owners']==[req for req,length in enumerate(expected) if length]
  assert row['full_current_block_rows_for_owned_requests']==8*len(row['request_owners'])
 assert owner287['status']=='read_only_original_path_geometry_and_page_envelope'
 assert owner287['rank_cohort_rows']==40 and owner287['rank_cohort_files']==40
 assert sum(row['cycles'] for row in owner287['summaries'])==11968
 assert all(len(row['geometry_signatures'])==1 and
            row['geometry_signatures'][0]['owner_full_update_rows']==16 and
            row['geometry_signatures'][0]['global_qsl']==list(range(0,97,8)) and
            len(row['owner_sets'])==1 and len(row['owner_sets'][0])==2
            for row in owner287['summaries'])
 assert all(not item['intersects'] for row in owner287['summaries']
            for item in row['compressed_owner_history_nonowner_new_page_intersection'])
 observed_ownership={
  'status':'original_path_geometry_observed_consumer_closure_unknown',
  'evidence':'Run287 original FULL Graph all8 x five cohorts, 11968 rank-cycles; 40/40 Runtime pass',
  'rank_cohort_files':40,'rank_cycles':11968,
  'global_query_rows_per_cycle':96,'local_query_rows_per_rank_cycle':12,
  'owner_request_slots_per_rank_cycle':2,'owner_full_update_rows_per_rank_cycle':16,
  'logical_nonowner_new_rows_per_rank_cycle':80,
  'owner_slots_by_rank':{str(rank):owner287['summaries'][rank*5]['owner_sets'][0] for rank in range(8)},
  'compressed_owner_history_vs_derived_nonowner_new_page_overlap':False,
  'confidence':'medium for frozen cohort geometry; low for removable physical traffic',
  'limits':'Derived page envelope, not actual native scatter/state/DSpark/prefix consumer closure or an E2E saving'}
 owner294=[json.loads((ROOT/f'evidence/20260926_loop066_owner/run294/fixture/rank{rank}.json').read_text()) for rank in range(8)]
 owner295=[json.loads((ROOT/f'evidence/20260926_loop066_owner/run295/fixture/rank{rank}.json').read_text()) for rank in range(8)]
 owner296=[json.loads((ROOT/f'evidence/20260926_loop066_owner/run296/fixture/rank{rank}.json').read_text()) for rank in range(8)]
 owner297=json.loads((ROOT/'evidence/20260926_loop066_owner/run297/timing_analysis.json').read_text())
 owner298=json.loads((ROOT/'evidence/20260926_loop066_owner/run298/graph_analysis.json').read_text())
 owner_alias=json.loads((ROOT/'evidence/20260926_loop066_owner/alias_census_run287.json').read_text())
 assert all(row['rank']==rank and row['A_B_owner_output']['matched_values_exact'] and
            row['A_B_owner_output']['other_valid_slots']==4 and
            row['A_A_output']['matched_values_exact'] and
            row['A_after_B_output']['matched_values_exact']
            for group in (owner294,owner295) for rank,row in enumerate(group))
 assert all(row['owner_state_delta']['B_changed_bytes_different_in_A']==0 for row in owner295)
 assert all(row['rank']==rank and row['stage']=='complete' and row['gate_pass'] and
            row['qli_output_shape']==[12,1,512] and row['qli_A_B_exact'] and
            row['owner_typed_slot']['key_scale_exact'] and
            all(domain['A_B_exact'] for domain in row['state_domains'].values()) and
            all(domain['A_B_exact'] for domain in row['post_scatter_state_domains'].values())
            for rank,row in enumerate(owner296))
 assert owner297['status']=='valid_private_eager_event_span' and len(owner297['ranks'])==8
 assert all(len(row['pairs'])==5 for row in owner297['ranks'])
 assert owner298['status']=='valid_private_graph_replay' and len(owner298['ranks'])==8
 assert owner298['aggregate']['pairs']==80
 assert owner_alias['summary']['rank_cohort_files']==40 and owner_alias['summary']['unique_storage_layout_patterns']==1
 observed_ownership['private_layer2_owner16_fixture']={
  'status':'same_prestate_layer2_indexer_graph_exact_local_gain_unresolved_whole_producer_lifetime_open',
  'output_exact_ranks_run294':8,'output_exact_ranks_run295':8,
  'whole_historical_owner_page_state_exact_ranks_run295':sum(row['A_B_owner_state_bytes_exact'] for row in owner295),
  'B_prestate_changed_bytes_different_in_A_by_rank':[row['owner_state_delta']['B_changed_bytes_different_in_A'] for row in owner295],
  'A_only_prestate_changed_bytes_in_owner_table_pages_by_rank':[row['owner_state_delta']['A_only_changed_bytes'] for row in owner295],
  'first_difference_nonowner_current_write_alias_ranks':7,
  'state_current_write_read_domain_exact':True,
  'state_domain_exact_ranks_run296':8,
  'typed_key_scale_owner_slot_exact_ranks_run296':8,
  'native_QLI_topk_exact_ranks_run296':8,
  'private_eager_native_chain_run297':{
   'parity_ranks':8,'paired_triplets_per_rank':5,
   'update_B_minus_full_rank_median_ms':owner297['aggregate']['update_ms']['all_rank_paired_B_minus_full_median_ms'],
   'through_QLI_B_minus_full_rank_median_ms':owner297['aggregate']['through_qli_ms']['all_rank_paired_B_minus_full_median_ms'],
   'through_QLI_B_faster_than_both_pairs':owner297['aggregate']['through_qli_ms']['all_rank_both_control_pair_wins'],
   'scope':'eager event span may include Host launch gaps; no stable local benefit, not Graph/Hardware/Product bound'},
  'private_graph_chain_run298':{
   'parity_ranks':8,'paired_triplets_per_rank':10,
   'pooled_B_minus_full_replay_ms':owner298['aggregate']['pooled_paired_B_minus_full_median_ms'],
   'pooled_A_A2_abs_drift_ms':owner298['aggregate']['pooled_A_A2_abs_drift_median_ms'],
   'B_faster_than_both_pairs':owner298['aggregate']['B_faster_than_both_pairs'],
   'scope':'private one-layer Graph replay, small signal below control drift; no Product gain'},
  'whole_producer_alias_census_run287':{
   'rank_cohort_files':40,
   'unique_storage_layout_patterns':1,
   'relevant_unique_backing_bytes_per_rank':owner_alias['summary']['total_unique_backing_bytes_per_rank_cohort'][0],
   'relevant_backing_count':len(owner_alias['summary']['representative_backings']),
   'scope':'three layer2-related backings include cross-layer aliases; byte-level liveness and Draft views unknown'},
  'Sparse_and_cross_cycle_lifetime_closed':False,
  'exposed_target_cycle_ms':None,'formal_E2E_gain_tps':None,
  'scope':'private one-call eager c4 indexer at real layer2 through typed scatter and native QLI; not compulsory-byte or exposed-wall estimate',
  'evidence':'Run294–298 fixtures and paired analyses, Run287 alias census and Astra independent review'}
 assert compressor['status']=='valid' and counters['status']=='valid' and hccl['status']=='valid'
 gmm=inventory['target']['gmm']
 latest=[w for w in counters['windows'] if w['capture']==counters['latest_capture']]
 assert len(latest)==16
 def counter_median(key):
  return statistics.median(sum(f[key] for name,f in w['families'].items()
                               if name!='communication')*1024/1e9 for w in latest)
 partial_resource={
  'canonical_matmul_arithmetic_Gflop_per_rank_target_cycle':{
   'routed_GMM_conditional_route_sample':gmm['logical_matmul_flops_per_rank_cycle']['value']/1e9,
   'compressor_fixed_shapes':sum(v['nominal_two_projection_flops_per_cycle']
                                 for v in compressor['summary'].values())/1e9,
   'scope':'partial known operators only; not a proof of minimum possible algorithmic operations',
   'evidence':'Run242 GMM route/shape inventory; Run256 exact Compressor shape count'},
  'parameter_tensor_footprint_GB_per_rank_target_cycle':{
   'routed_GMM_active_packed_cross_sample':gmm['packed_active_weight_bytes_per_rank_cycle']['value']/1e9,
   'compressor_two_BF16_weights':sum(v['unique_two_weight_tensor_footprint_bytes_per_cycle']
                                     for v in compressor['summary'].values())/1e9,
   'scope':'distinct tensor footprint screen, not compulsory HBM read bytes',
   'evidence':'Run242 and Run256'},
  'current_FULL_Graph_counter_GB_per_rank_target_cycle':{
   'read_median_whole_window':counter_median('read_KB'),
   'write_median_whole_window':counter_median('write_KB'),
   'read_sum_of_family_medians':sum(v['read_KB']['median'] for name,v in counters['summary'].items() if name!='communication')*1024/1e9,
   'write_sum_of_family_medians':sum(v['write_KB']['median'] for name,v in counters['summary'].items() if name!='communication')*1024/1e9,
   'scope':'Run247 observed task counters, not compulsory traffic; median of sums differs from sum of family medians'},
  'HCCL_reported_operation_payload_bytes_per_rank_target_cycle':hccl['latest_reported_payload_bytes_per_rank_cycle'],
  'conditional_CP_request_ownership_screen':ownership,
  'observed_CP_request_ownership_run287':observed_ownership,
  'unknown':'full target/prefill/DSpark necessary arithmetic, unique HBM/KV traffic, physical link bytes and concurrent attainable capacity'}
 # Observed Product wall decomposition is descriptive, not an additive bound:
 # latest-rank cohort runtime and client wall use different clock boundaries.
 current_formal_envelope=[]
 run99=ROOT/'evidence/20260924_loop036_metadata/run99'
 for repeat,cohorts in enumerate((range(5,9),range(9,13),range(13,17)),1):
  bench=json.loads((run99/f'bench48_{repeat}.json').read_text())['summary']
  cohort_rows=[]
  for cohort in cohorts:
   ranks=[json.loads((run99/'runtime'/f'rank{rank}_cohort{cohort}.json').read_text()) for rank in range(8)]
   assert all(row['pass'] and row['target_graph_mode']=='FULL' for row in ranks)
   cycle_count=max(row['cycles'] for row in ranks)
   max_wall=max(row['wall_seconds'] for row in ranks)
   cohort_rows.append({'cohort':cohort,'cycles':cycle_count,
                       'max_rank_runtime_wall_s':max_wall,
                       'observed_runtime_wall_ms_per_cycle':1000*max_wall/cycle_count})
  runtime_envelope=sum(row['max_rank_runtime_wall_s'] for row in cohort_rows)
  current_formal_envelope.append({'repeat':repeat,'client_wall_s':bench['duration_s'],
    'client_output_tps':bench['output_tps'],'cohorts':cohort_rows,
    'sum_cohort_max_rank_runtime_wall_s':runtime_envelope,
    'client_minus_runtime_envelope_s':bench['duration_s']-runtime_envelope,
    'scope':'descriptive cross-clock residual; not proven removable and not an additive timing model'})
 unknown_keys=set(costs)-set(NODES)
 if unknown_keys:raise ValueError(f'unknown node costs: {unknown_keys}')
 cp_unknown=set(cp_costs)-set(CP_C4_NODES)
 if cp_unknown:raise ValueError(f'unknown CP c4 node costs: {cp_unknown}')
 out={'schema_version':1,'contract':'DeepSeek V4 W4A8, 8x910B3, DP1TP8, DSpark7, 48x32K->1024 c12',
  'current_formal_tps':571.681,'algorithmic_token_accounting':{'total_useful_output_tokens':49152,'max_target_outputs_per_request_cycle':8,'max_concurrent_requests':12,'absolute_min_target_cycles':512,'current_Run99_formal_cycles':[1217,1212,1206],'lower_bound_formula':'sum over cohorts max_i ceil((1024-initial_output_count_ci)/8)','Run99_initial_output_counts_per_slot':0,'scope':'loose DSpark7/Target8 acceptance cardinality limit, not achievable acceptance or latency/TPS bound','evidence':'frozen 48x1024 c12 contract; FixedDecodeConfig and Run99'},'candidate_formal_observation':{'tps':594.1334346560858,'control_tps':582.8518793259907,'status':'exposed_gain_not_established_Run268'},
  'bounds':{'current':{'formal_tps':571.681,'source':'Run99 three formal repeats, 48x32K->1024 c12, 8-rank FULL Graph gates','status':'measured'},'hardware_resource':{'physical_capacity_lower_bound_ms':None,'attainable_engineering_scenario_ms':None,'status':'unknown_missing_compulsory_work_and_full_graph_capacity'},'scheduling_aware':{'dependency_relaxation_ms':None,'resource_constrained_achievable_ms':None,'status':'unknown_missing_node_costs_resource_contention_and_alias_proof'},'product_e2e':{'tps_upper_bound':None,'status':'unknown_missing_prefill_trajectory_acceptance_and_serving_edges'}},
  'nodes':NODES,'node_cost_ledger':node_cost_ledger,'semantic_edges':[e for e in OVERLAP if e[2] in ('semantic_RAW','KV_state_RAW')],
  'partial_resource_inventory':partial_resource,
  'observed_request_progress':{'run287_instrumented_trajectory':parked['aggregate'],
    'scope':'slot-cycle capacity accounting only; no necessary FLOPs, traffic or TPS deduction',
    'evidence':'Run287 all8 ownership JSONL; independent Astra acceptance screen'},
  'current_formal_observed_wall_envelope':current_formal_envelope,
  'product_handoff_gate_diagnostic':{'ready_cohorts':len(handoff['ready_cohorts']),'rank_cohort_pass':handoff['rank_cohort_pass'],'sample_reason_counts':handoff['sample_reason_counts'],'scope':'Run283 original CP, rank0 Host instrumentation, not formal throughput or proof of candidate coverage'},
  'current_edges':CURRENT,'metadata_overlap_edges':OVERLAP,
  'cp_c4_target':{'source':'frozen enable_dsa_cp: context_parallel/dsa_cp.py:1400-1805',
    'nodes':CP_C4_NODES,'semantic_edges':CP_C4_SEMANTIC,
    'current_edges':CP_C4_CURRENT,'fork_edges':CP_C4_FORK,
    'hidden_gather_edges':CP_C4_HIDDEN_GATHER,
    'run292_one_layer_hidden_gather':{'status':'device_overlap_proven_endpoint_unresolved',
      'selected_layer':'Target layer2 decode96 local BF16 [12,4096] -> gathered [96,4096]',
      'all_rank_cycle_samples':16,'overlap_samples':16,
      'median_hccl_gather_device_us':14.88,'median_device_overlap_with_local_Q_us':13.75,
      'matched_A1_profile':'Run293 complete; all8 x two cycles, independent capture',
      'paired_rank_cycle_WKV_advance_median_us':hidden_matched['summary']['pre_to_wkv_us']['paired_delta_median'],
      'paired_rank_cycle_next_AllToAll_end_advance_median_us':hidden_matched['summary']['pre_to_a2a_end_us']['paired_delta_median'],
      'target_tail_B_faster_samples':hidden_matched['summary']['pre_to_target_graph_end_us']['B_smaller_samples'],
      'target_tail_samples':hidden_matched['summary']['pre_to_target_graph_end_us']['samples'],
      'same_state_typed_parity':False,'whole_cycle_gain_proven':False,'formal_E2E_gain_proven':False,
      'evidence':'Run292/293 latest-per-rank matched traces, independent Astra task-id metrics'},
    'fine_semantic_nodes':CP_C4_FINE_NODES,'fine_semantic_edges':CP_C4_FINE_SEMANTIC,
    'fine_semantic_dag':analyse(CP_C4_FINE_SEMANTIC,{},CP_C4_FINE_NODES,'output'),
    'run282_single_layer_overlap_screen':{'aux_compressor_median_us':65.98,'cross_stream_overlap_median_us':65.00,'historical_first_c4_rotary_median_us':5.67,'concurrent_rotary_median_us':64.19,'status':'actual_overlap_no_exposed_saving_proven','matched_control':{'run':'Run284 immediate same profiler, not original A0','paired_rank_cycle_next_AllToAll_delta_median_us':-39.875,'samples':16,'improved_samples':16,'same_state':False,'whole_cycle_gain_proven':False,'formal_E2E_gain_proven':False,'note':'immediate reorders original QLI->main Compressor to main Compressor->join->QLI'},'evidence':'Run282/284 matched traces and Astra High independent review'},
    'run285_eager_numerical_gate':{'status':'inconclusive','cohort_clients_exact1024':12,'ranks_with_five_pass_rows':8,'modes':['off','overlap','off','immediate','off'],'off0_vs_off2_argmax_equal':96,'off0_vs_off4_argmax_equal':94,'overlap_vs_off0_argmax_equal':93,'note':'later off follows candidate, so divergence may be self-replay noise or uncaptured persistent effects; mixed cache-view max-abs invalid as layer2-only error','product_gain_claim':False,'evidence':'Run285 pages/analysis and Astra High numerical review'},
    'current_dag':analyse(CP_C4_CURRENT,cp_costs,CP_C4_NODES,'output'),
    'fork_dag':analyse(CP_C4_FORK,cp_costs,CP_C4_NODES,'output'),
    'hidden_gather_dag':analyse(CP_C4_HIDDEN_GATHER,cp_costs,CP_C4_NODES,'output'),
    'scope':'coarse current/fork graphs model restricted implemented schedules; fine semantic graph removes false qr/Q and query/QLI barriers; all need same-path cost, contention and all-rank join evidence',
    'evidence':'Run276 Astra High source audit'},
  'symbolic_constraints':{'resource_physical_floor':'max over shared resources of compulsory work / defensible hardware capacity UPPER bound; an observed attained bandwidth is not automatically such an upper bound','resource_attainable_scenario':'use same-shape concurrent capacity measurements to build an executable resource allocation; empirical attained capacity alone is not a strict latency lower bound','scheduling_relaxation':'max(physical resource floor, legal dependency critical path with collective rendezvous); resource-constrained executable makespan also includes interference, streams, storage and event overhead','product_throughput':'49152 / finite frozen client wall time after prefill, decode, parking, useful tokens/cycle and publication DAG; no additive phase shortcut'},
  'current_dag':analyse(CURRENT,costs),'overlap_dag':analyse(OVERLAP,costs),
  'already_measured_removable_node':{'name':'DSpark-unused MTP stash','necessary_math_flops':0,'necessary_bytes':0,'current_reported_read_GB':0.537537728,'current_reported_write_GB':0.537424896,'current_task_sum_ms':0.7993415,'causal_candidate_copies_per_cycle':0,'exposed_ms':None,'evidence':'Run257, Run260-263'},
  'unresolved':['current per-node same-path timings with async join','Run275 candidate-consumed 8-rank continuous metadata/alias gate passed; verification-free A0/B timing and hidden native workspace remain open','necessary target+DSpark FLOPs/unique traffic','8-rank FULL Graph compute/HBM/HCCL attainable capacity','HCCL physical link bytes/arrival/overlap','CP c4 indexer/main compressor same-path service costs, Graph concurrency and resource contention','Run294/295 owner16 private Compressor outputs exact all8 and B-changed bytes match A; full current write/live8 read domain, typed scatter/QLI/Sparse, prefix/lifetime and exposed cycle remain open before interpreting replica updates as removable compulsory work','DSpark useful tokens/cycle upper limit','prefill/admission/output product trajectory','Run281 only 2/5 expected fixed cohorts; original-path Run283 had 5/5 and exact gate shape trajectory, cross-run cause remains unknown','one-layer CP fork eager A0/overlap/A0 Run285 inconclusive; local fork-entry typed numerical gate remains open before FULL Graph/all21 promotion','decode hidden AllGather/local Q semantic overlap requires FULL Graph capture/replay validation'],
  'limits':['A DAG with isolated duration inputs is only an optimistic no-contention relaxation, not a hardware ceiling.','Current graph edges include implementation ordering; metadata_overlap_edges require private scratch and parking invalidation.','Temporal profiler task sums and AIC+AIV bytes are not compulsory work or exposed wall time.','No TPS result is emitted from an incomplete dependency or capacity model.']}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'current_dag':out['current_dag'],'overlap_dag':out['overlap_dag'],'bounds':out['bounds']},indent=2))
if __name__=='__main__':main()
