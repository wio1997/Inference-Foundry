from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,gzip,socket,statistics,math
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030');R=A/'observer-20261030T160315Z-76b79f0f/value-review'
OUT=A/'profile-readonly-20261031/EP-routes-ownership-and-placement-tradeoffs-v17.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
distribution=pin(A/'profile-readonly-20261031/EP-row-distribution-and-placement-counterfactual-v17.json','4dd1ffc5aaf63d7ad54bcdd23f1a9435962a319976d7c9df8ecd7b5bde1b1c69')
raw=(R/'decoded-records.jsonl.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029';records=[json.loads(l)for l in gzip.decompress(raw).decode().splitlines()]
groups=defaultdict(list)
for r in records:groups[(r['role'],r['layer'],r['ordinal'])].append(r)
def stats(v):
 if not v:return None
 v=sorted(v);return dict(n=len(v),minimum=v[0],median=statistics.median(v),mean=statistics.mean(v),maximum=v[-1])
def matrix(rs,E,owner=None):
 out=[[0]*16 for _ in range(16)];expert=[0]*(16*E)
 for r in rs:
  for name,n in r['sender_expert_histogram'].items():
   i=int(name);assert 0<=i<len(expert);expert[i]+=n;dest=i//E if owner is None else owner[i];out[r['ep_rank']][dest]+=n
 return out,expert
def features(m):
 total=sum(map(sum,m));local=sum(m[i][i]for i in range(16));cross=sum(m[i][j]for i in range(16)for j in range(16)if i//8!=j//8)
 return dict(total_assignments=total,local_assignments=local,remote_assignments=total-local,cross_TP8_partition_assignments=cross,local_fraction=local/total if total else None,cross_TP8_partition_fraction=cross/total if total else None,nonzero_directed_sender_receiver_pairs=sum(n>0 for row in m for n in row),max_sender_destination_degree=max(sum(n>0 for n in row)for row in m))
formal=defaultdict(list)
for c in distribution['cohorts']:
 if c['phase']=='formal_C8_measure':formal[(c['role'],c['layer'])].append(c)
training={k:min(v,key=lambda c:c['ordinal'])for k,v in formal.items()}
rows=[];buffer_groups=defaultdict(list)
for c in distribution['cohorts']:
 key=(c['role'],c['layer'],c['ordinal']);rs=sorted(groups[key],key=lambda r:r['ep_rank']);E=c['E_local'];m,ex=matrix(rs,E)
 assert ex==c['receiver_expert_rows_by_logical_id']
 assert[sum(row)for row in m]==[c['K']*x for x in c['valid_sender_rows_by_rank']]
 assert[sum(m[i][j]for i in range(16))for j in range(16)]==c['receiver_rows_by_rank']
 for r in rs:buffer_groups[(r['role'],r['ep_rank'],r['counts_ptr'])].append(r)
 total=c['total_assignments'];maxexpert=c['maximum_expert_rows'];meanbound=math.ceil(total/16)
 one_replicated_bounds={}
 counts=c['receiver_expert_rows_by_logical_id'];hottest=max(range(len(counts)),key=lambda i:(counts[i],-i))
 for replicas in [2,4]:
  max_after=max(math.ceil(n/replicas)if i==hottest else n for i,n in enumerate(counts));one_replicated_bounds[replicas]=max(meanbound,max_after)
 row=dict(role=c['role'],layer=c['layer'],ordinal=c['ordinal'],phase=c['phase'],sender_receiver_assignment_matrix=m,current_routes=features(m),placement_strategy_values=sorted({r['placement_strategy']for r in rs}),redundant_expert_values=sorted({r['redundant_experts']for r in rs}),mask_domains=sorted({r['mask_domain']for r in rs}),capturing_entry_counts=dict(Counter(r['capturing_entry']for r in rs)),group_ranks=rs[0]['group_ranks'],all_group_binding_complete=all(r['group_binding_complete']for r in rs),current_max_rank_rows=c['current_maximum_rank_rows'],max_single_expert_rows=maxexpert,single_expert_bound_over_current_max=maxexpert/c['current_maximum_rank_rows']if c['current_maximum_rank_rows']else None,unreplicated_count_bound=max(meanbound,maxexpert),greedy_attains_independent_count_bound=c['greedy_per_cohort_assignment']['maximum']==c['independent_unreplicated_count_lower_bound'],one_hottest_expert_replicated_count_lower_bounds=one_replicated_bounds,replica_bound_qualification='Optimistic independent count bound only; extra physical weight slots/memory/ABI/locality/numerical scheduling not constructed.')
 if c['phase']=='formal_C8_measure':
  train=training[(c['role'],c['layer'])];owner=train['greedy_per_cohort_assignment']['owner_rank_by_logical_expert'];cm,ce=matrix(rs,E,owner);assert ce==ex
  row['first_formal_fixed_mapping']=dict(training_ordinal=train['ordinal'],is_training_sample=train['ordinal']==c['ordinal'],receiver_rows_by_rank=[sum(cm[i][j]for i in range(16))for j in range(16)],routes=features(cm),sender_receiver_assignment_matrix=cm)
 rows.append(row)
summarygroups=[]
for role,phase in sorted({(r['role'],r['phase'])for r in rows}):
 rs=[r for r in rows if(r['role'],r['phase'])==(role,phase)];held=[r for r in rs if r.get('first_formal_fixed_mapping')and not r['first_formal_fixed_mapping']['is_training_sample']]
 summarygroups.append(dict(role=role,phase=phase,cohorts=len(rs),actual_placement_strategies=sorted({v for r in rs for v in r['placement_strategy_values']}),actual_redundant_experts=sorted({v for r in rs for v in r['redundant_expert_values']}),current_local_assignment_fraction=stats([r['current_routes']['local_fraction']for r in rs if r['current_routes']['local_fraction']is not None]),current_cross_TP8_partition_fraction=stats([r['current_routes']['cross_TP8_partition_fraction']for r in rs if r['current_routes']['cross_TP8_partition_fraction']is not None]),max_expert_bound_over_current_rank_max=stats([r['single_expert_bound_over_current_max']for r in rs if r['single_expert_bound_over_current_max']is not None]),greedy_attains_count_bound=sum(r['greedy_attains_independent_count_bound']for r in rs),heldout_fixed_mapping_route_tradeoffs=dict(evaluations=len(held),local_fraction_change=stats([r['first_formal_fixed_mapping']['routes']['local_fraction']-r['current_routes']['local_fraction']for r in held]),cross_TP8_fraction_change=stats([r['first_formal_fixed_mapping']['routes']['cross_TP8_partition_fraction']-r['current_routes']['cross_TP8_partition_fraction']for r in held]),more_cross_TP8_assignments=sum(r['first_formal_fixed_mapping']['routes']['cross_TP8_partition_assignments']>r['current_routes']['cross_TP8_partition_assignments']for r in held),fewer_cross_TP8_assignments=sum(r['first_formal_fixed_mapping']['routes']['cross_TP8_partition_assignments']<r['current_routes']['cross_TP8_partition_assignments']for r in held))))
buffers=[dict(role=k[0],ep_rank=k[1],counts_ptr=k[2],samples=len(v),layers=sorted({r['layer']for r in v}),distinct_layers=len({r['layer']for r in v}),distinct_ordinals=len({r['ordinal']for r in v}))for k,v in buffer_groups.items()]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),cohorts=len(rows),sender_receiver_per_expert_conservation_checks=len(rows),groups=summarygroups,count_buffer_identities=len(buffers),count_buffers_covering_multiple_layers=sum(r['distinct_layers']>1 for r in buffers),count_buffer_layer_count_hist=dict(Counter(r['distinct_layers']for r in buffers)),no_new_source_device_or_inference_work=True,overall_complete=False,limits=['Expert IDs here are the IDs observed by native dispatch and receiver slot r*E+i; checkpoint-logical to native-physical mapping must be proved before applying a permutation. Field names inherited from the preceding count-only artifact are not that proof.','TP8 halves refer to the configured DP/TP partition within this16-member EP group,not independent hosts or measured physical links. Assignment fraction is not byte/packet count.','Counts and synthetic owner mappings are not timing or hardware utilization and are not joined to the later hostprof capture.','Replica lower bounds are optimistic count models; replicated hot weights require extra slots/memory and agreed token selection,expert ID remap,readiness and combine order.','counts_ptr identity is keyed by role and EP rank; pointer reuse is not a native event-generation proof.'])
payload=(json.dumps(dict(summary=summary,cohorts=rows,reused_count_buffers=buffers),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,buffer_samples=buffers[:2],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
