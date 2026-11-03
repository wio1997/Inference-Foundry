from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib,statistics,math,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
OUT=A/'EP-partition-constrained-placement-counterfactual-v17.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
d=pin(A/'EP-row-distribution-and-placement-counterfactual-v17.json','4dd1ffc5aaf63d7ad54bcdd23f1a9435962a319976d7c9df8ecd7b5bde1b1c69');r=pin(A/'EP-routes-ownership-and-placement-tradeoffs-v17.json','b9ded553f181b708eff60eb308e5813d70110a08fba200d94cc45fbd813d9db3');route={(c['role'],c['layer'],c['ordinal']):c for c in r['cohorts']}
formal=defaultdict(list)
for c in d['cohorts']:
 if c['phase']=='formal_C8_measure':formal[(c['role'],c['layer'])].append(c)
def constrained(counts,E):
 owner=[None]*len(counts);loads=[0]*16;slots=[0]*16
 for half in range(2):
  experts=range(half*8*E,(half+1)*8*E)
  for e in sorted(experts,key=lambda e:(-counts[e],e)):
   rank=min((k for k in range(half*8,(half+1)*8)if slots[k]<E),key=lambda k:(loads[k],slots[k],k));owner[e]=rank;loads[rank]+=counts[e];slots[rank]+=1
 assert slots==[E]*16 and all(owner[e]//8==e//E//8 for e in range(len(counts)))
 return owner
rows=[];mappings=[]
for (role,layer),cs in sorted(formal.items()):
 cs=sorted(cs,key=lambda c:c['ordinal']);train=cs[0];E=train['E_local'];owner=constrained(train['receiver_expert_rows_by_logical_id'],E);mappings.append(dict(role=role,layer=layer,training_ordinal=train['ordinal'],E_local=E,owner_rank_by_observed_native_expert_ID=owner))
 for c in cs[1:]:
  rt=route[(role,layer,c['ordinal'])];counts=c['receiver_expert_rows_by_logical_id'];loads=[0]*16
  for e,n in enumerate(counts):loads[owner[e]]+=n
  assert sum(loads)==c['total_assignments']
  by_sender_to_original_half=[[sum(rt['sender_receiver_assignment_matrix'][s][j]for j in range(h*8,(h+1)*8))for h in range(2)]for s in range(16)]
  assert all(owner[e]//8==e//E//8 for e in range(len(counts)))
  halfbound=max([max(counts)]+[math.ceil(sum(counts[h*8*E:(h+1)*8*E])/8)for h in range(2)])
  global_fixed_max=max(rt['first_formal_fixed_mapping']['receiver_rows_by_rank']);assert max(loads)>=halfbound
  rows.append(dict(role=role,layer=layer,training_ordinal=train['ordinal'],evaluation_ordinal=c['ordinal'],current_maximum_rank_rows=c['current_maximum_rank_rows'],global_fixed_mapping_maximum_rank_rows=global_fixed_max,within_TP8_fixed_mapping_maximum_rank_rows=max(loads),within_TP8_receiver_rows_by_rank=loads,within_TP8_unreplicated_count_bound=halfbound,current_over_within_TP8_max=c['current_maximum_rank_rows']/max(loads)if max(loads)else None,within_TP8_over_global_fixed_max=max(loads)/global_fixed_max if global_fixed_max else None,cross_TP8_assignment_count_preserved=True,source_sender_to_destination_half_assignment_counts=by_sender_to_original_half))
def stats(v):
 if not v:return None
 v=sorted(v);return dict(n=len(v),minimum=v[0],median=statistics.median(v),mean=statistics.mean(v),maximum=v[-1])
groups=[]
for role in sorted({x['role']for x in rows}):
 rs=[x for x in rows if x['role']==role];groups.append(dict(role=role,evaluation_samples=len(rs),improved_count_max=sum(x['within_TP8_fixed_mapping_maximum_rank_rows']<x['current_maximum_rank_rows']for x in rs),same_count_max=sum(x['within_TP8_fixed_mapping_maximum_rank_rows']==x['current_maximum_rank_rows']for x in rs),worse_count_max=sum(x['within_TP8_fixed_mapping_maximum_rank_rows']>x['current_maximum_rank_rows']for x in rs),current_over_within_TP8_fixed_max=stats([x['current_over_within_TP8_max']for x in rs if x['current_over_within_TP8_max']is not None]),within_TP8_over_global_fixed_max=stats([x['within_TP8_over_global_fixed_max']for x in rs if x['within_TP8_over_global_fixed_max']is not None]),cross_TP8_assignment_counts_preserved=sum(x['cross_TP8_assignment_count_preserved']for x in rs)))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),training_mappings=len(mappings),heldout_saved_count_samples=len(rows),groups=groups,fixed_slots_per_rank=True,no_redundant_expert_added=True,no_retraining_on_evaluation_samples=True,no_new_source_model_or_device_work=True,overall_complete=False,limits=['Within-TP8 owner permutation preserves native expert-ID destination half by construction and therefore preserves cross-half assignment counts,not per-link bytes or timing.','TP8 halves are configured rank partitions,not a complete physical topology model. Within-half traffic and on-rank locality can change.','Count-only feasible mappings do not prove supported native expert-ID/weight mapping ABI or numeric equality.','First-formal training and later count sample evaluation are from one observer run; no end-to-end runtime validation or measured speedup.'])
payload=(json.dumps(dict(summary=summary,mappings=mappings,evaluations=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
