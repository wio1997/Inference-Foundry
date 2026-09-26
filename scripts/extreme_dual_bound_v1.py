#!/usr/bin/env python3
"""Executable dependency skeleton for the frozen Extreme cycle.

Only supplied, same-path node costs produce a numeric critical-path relaxation.
It never promotes standalone kernel time, trace bytes, or V0 assumptions to a
hardware ceiling. Resource-constrained scheduling and Product E2E remain open.
"""
import argparse, json
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
COMMON=[('target_k','acceptance_k','semantic_RAW'),('target_k','dspark_old_state_k','semantic_RAW'),('acceptance_k','state_advance_k','semantic_RAW'),('state_advance_k','dspark_old_state_k','implementation_order'),('dspark_old_state_k','dspark_model_k','semantic_RAW'),('dspark_model_k','next_ids_k1','semantic_RAW'),('next_ids_k1','target_k1','semantic_RAW'),('target_k','target_k1','KV_state_RAW')]
CURRENT=COMMON+[('dspark_model_k','metadata_k1','implementation_order'),('state_advance_k','metadata_k1','semantic_RAW'),('metadata_k1','target_k1','semantic_RAW')]
OVERLAP=COMMON+[('state_advance_k','metadata_k1','semantic_RAW'),('metadata_k1','metadata_commit_k1','semantic_RAW'),('dspark_model_k','metadata_commit_k1','storage_WAR_guard'),('metadata_commit_k1','target_k1','semantic_RAW')]

def analyse(edges,costs):
 active={node for src,dst,_ in edges for node in (src,dst)}
 incoming={k:[] for k in active}
 for src,dst,kind in edges:
  assert src in NODES and dst in NODES and kind in {'semantic_RAW','KV_state_RAW','storage_WAR_guard','implementation_order'}
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
 return {'topological_order':order,'critical_path_ms':finish['target_k1'],'missing_node_costs':[]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--durations-json');a=ap.parse_args()
 costs=json.loads(Path(a.durations_json).read_text()) if a.durations_json else {}
 unknown_keys=set(costs)-set(NODES)
 if unknown_keys:raise ValueError(f'unknown node costs: {unknown_keys}')
 out={'schema_version':1,'contract':'DeepSeek V4 W4A8, 8x910B3, DP1TP8, DSpark7, 48x32K->1024 c12',
  'current_formal_tps':571.681,'latest_candidate_formal_tps_pending_control':594.1334346560858,
  'bounds':{'hardware_resource':{'time_lower_bound_ms':None,'status':'unknown_missing_compulsory_work_and_full_graph_capacity'},'scheduling_aware':{'time_lower_bound_ms':None,'status':'unknown_missing_node_costs_resource_contention_and_alias_proof'},'product_e2e':{'tps_upper_bound':None,'status':'unknown_missing_prefill_trajectory_acceptance_and_serving_edges'}},
  'nodes':NODES,'semantic_edges':[e for e in OVERLAP if e[2] in ('semantic_RAW','KV_state_RAW')],
  'current_edges':CURRENT,'metadata_overlap_edges':OVERLAP,
  'symbolic_constraints':{'resource_time':'max over shared resources of compulsory work / proven concurrent upper envelope capacity; capacities and compulsory work currently null','scheduling_time':'max(resource_time, dependency critical path under legal buffers and collective rendezvous); no free overlap when node service times change','product_throughput':'49152 / finite frozen client wall time after prefill, decode, parking, useful tokens/cycle and publication DAG; no additive phase shortcut'},
  'current_dag':analyse(CURRENT,costs),'overlap_dag':analyse(OVERLAP,costs),
  'already_measured_removable_node':{'name':'DSpark-unused MTP stash','necessary_math_flops':0,'necessary_bytes':0,'current_reported_read_GB':0.537537728,'current_reported_write_GB':0.537424896,'current_task_sum_ms':0.7993415,'causal_candidate_copies_per_cycle':0,'exposed_ms':None,'evidence':'Run257, Run260-263'},
  'unresolved':['current per-node same-path timings with async join','same-state scratch/commit alias and parking legality','necessary target+DSpark FLOPs/unique traffic','8-rank FULL Graph compute/HBM/HCCL attainable capacity','HCCL physical link bytes/arrival/overlap','DSpark useful tokens/cycle upper limit','prefill/admission/output product trajectory'],
  'limits':['A DAG with isolated duration inputs is only an optimistic no-contention relaxation, not a hardware ceiling.','Current graph edges include implementation ordering; metadata_overlap_edges require private scratch and parking invalidation.','Temporal profiler task sums and AIC+AIV bytes are not compulsory work or exposed wall time.','No TPS result is emitted from an incomplete dependency or capacity model.']}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'current_dag':out['current_dag'],'overlap_dag':out['overlap_dag'],'bounds':out['bounds']},indent=2))
if __name__=='__main__':main()
