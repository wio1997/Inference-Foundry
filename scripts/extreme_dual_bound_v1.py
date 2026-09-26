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
 costs=json.loads(Path(a.durations_json).read_text()) if a.durations_json else {}
 cp_costs=json.loads(Path(a.cp_durations_json).read_text()) if a.cp_durations_json else {}
 inventory=json.loads((ROOT/'evidence/20260926_loop060_resource/run242/inventory.json').read_text())
 compressor=json.loads((ROOT/'evidence/20260926_loop062_nongmm/run256/census.json').read_text())
 counters=json.loads((ROOT/'evidence/20260926_loop060_resource/run247/analysis.json').read_text())
 hccl=json.loads((ROOT/'evidence/20260926_loop061_bound/run250/payload.json').read_text())
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
  'unknown':'full target/prefill/DSpark necessary arithmetic, unique HBM/KV traffic, physical link bytes and concurrent attainable capacity'}
 unknown_keys=set(costs)-set(NODES)
 if unknown_keys:raise ValueError(f'unknown node costs: {unknown_keys}')
 cp_unknown=set(cp_costs)-set(CP_C4_NODES)
 if cp_unknown:raise ValueError(f'unknown CP c4 node costs: {cp_unknown}')
 out={'schema_version':1,'contract':'DeepSeek V4 W4A8, 8x910B3, DP1TP8, DSpark7, 48x32K->1024 c12',
  'current_formal_tps':571.681,'algorithmic_token_accounting':{'total_useful_output_tokens':49152,'max_target_outputs_per_request_cycle':8,'max_concurrent_requests':12,'absolute_min_target_cycles':512,'current_Run99_formal_cycles':[1217,1212,1206],'scope':'loose DSpark7/Target8 acceptance cardinality limit, not achievable acceptance or latency/TPS bound','evidence':'frozen 48x1024 c12 contract; FixedDecodeConfig and Run99'},'candidate_formal_observation':{'tps':594.1334346560858,'control_tps':582.8518793259907,'status':'exposed_gain_not_established_Run268'},
  'bounds':{'hardware_resource':{'physical_capacity_lower_bound_ms':None,'attainable_engineering_scenario_ms':None,'status':'unknown_missing_compulsory_work_and_full_graph_capacity'},'scheduling_aware':{'dependency_relaxation_ms':None,'resource_constrained_achievable_ms':None,'status':'unknown_missing_node_costs_resource_contention_and_alias_proof'},'product_e2e':{'tps_upper_bound':None,'status':'unknown_missing_prefill_trajectory_acceptance_and_serving_edges'}},
  'nodes':NODES,'semantic_edges':[e for e in OVERLAP if e[2] in ('semantic_RAW','KV_state_RAW')],
  'partial_resource_inventory':partial_resource,
  'current_edges':CURRENT,'metadata_overlap_edges':OVERLAP,
  'cp_c4_target':{'source':'frozen enable_dsa_cp: context_parallel/dsa_cp.py:1400-1805',
    'nodes':CP_C4_NODES,'semantic_edges':CP_C4_SEMANTIC,
    'current_edges':CP_C4_CURRENT,'fork_edges':CP_C4_FORK,
    'current_dag':analyse(CP_C4_CURRENT,cp_costs,CP_C4_NODES,'output'),
    'fork_dag':analyse(CP_C4_FORK,cp_costs,CP_C4_NODES,'output'),
    'scope':'one c4 layer relaxation only; requires measured same-path service costs, stream overhead, shared resource contention and all-rank join before E2E inference',
    'evidence':'Run276 Astra High source audit'},
  'symbolic_constraints':{'resource_physical_floor':'max over shared resources of compulsory work / defensible hardware capacity UPPER bound; an observed attained bandwidth is not automatically such an upper bound','resource_attainable_scenario':'use same-shape concurrent capacity measurements to build an executable resource allocation; empirical attained capacity alone is not a strict latency lower bound','scheduling_relaxation':'max(physical resource floor, legal dependency critical path with collective rendezvous); resource-constrained executable makespan also includes interference, streams, storage and event overhead','product_throughput':'49152 / finite frozen client wall time after prefill, decode, parking, useful tokens/cycle and publication DAG; no additive phase shortcut'},
  'current_dag':analyse(CURRENT,costs),'overlap_dag':analyse(OVERLAP,costs),
  'already_measured_removable_node':{'name':'DSpark-unused MTP stash','necessary_math_flops':0,'necessary_bytes':0,'current_reported_read_GB':0.537537728,'current_reported_write_GB':0.537424896,'current_task_sum_ms':0.7993415,'causal_candidate_copies_per_cycle':0,'exposed_ms':None,'evidence':'Run257, Run260-263'},
  'unresolved':['current per-node same-path timings with async join','Run275 candidate-consumed 8-rank continuous metadata/alias gate passed; verification-free A0/B timing and hidden native workspace remain open','necessary target+DSpark FLOPs/unique traffic','8-rank FULL Graph compute/HBM/HCCL attainable capacity','HCCL physical link bytes/arrival/overlap','CP c4 indexer/main compressor same-path service costs, Graph concurrency and resource contention','DSpark useful tokens/cycle upper limit','prefill/admission/output product trajectory'],
  'limits':['A DAG with isolated duration inputs is only an optimistic no-contention relaxation, not a hardware ceiling.','Current graph edges include implementation ordering; metadata_overlap_edges require private scratch and parking invalidation.','Temporal profiler task sums and AIC+AIV bytes are not compulsory work or exposed wall time.','No TPS result is emitted from an incomplete dependency or capacity model.']}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'current_dag':out['current_dag'],'overlap_dag':out['overlap_dag'],'bounds':out['bounds']},indent=2))
if __name__=='__main__':main()
