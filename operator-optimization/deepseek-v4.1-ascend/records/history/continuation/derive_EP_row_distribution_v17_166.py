from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,gzip,socket,statistics,math
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
R=A/'observer-20261030T160315Z-76b79f0f/value-review';OUT=A/'profile-readonly-20261031/EP-row-distribution-and-placement-counterfactual-v17.json';assert not OUT.exists()
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
phase=pin(R/'snapshot-phase-index.json','677d29c367ae175e1161681db6b0f3a470bf96d2a39d41c7da5445734beaa631');phase={p['index']:p for p in phase}
cohortref=pin(R/'cohorts.json','ffab97e34745f7cee137c9464823f14e1b436fb9fad0c977144e58422a3a88e7')
raw=(R/'decoded-records.jsonl.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029'
records=[json.loads(l)for l in gzip.decompress(raw).decode().splitlines()];assert len(records)==4736
grouped=defaultdict(list)
for r in records:
 p=phase[r['index']];assert(r['role'],r['layer'],r['ordinal'],r['ep_rank'])==(p['role'],p['layer'],p['ordinal'],p['rank'])
 assert r['receiver_count_sum']==sum(r['receiver_expert_counts']);grouped[(r['role'],r['layer'],r['ordinal'])].append((r,p['phase']))
def stats(xs):
 if not xs:return None
 xs=sorted(xs);return dict(n=len(xs),minimum=xs[0],median=statistics.median(xs),mean=statistics.mean(xs),maximum=xs[-1])
def greedy(counts,ranks,slots):
 loads=[0]*ranks;bins=[[]for _ in range(ranks)];owner=[None]*len(counts)
 for expert in sorted(range(len(counts)),key=lambda i:(-counts[i],i)):
  rank=min((r for r in range(ranks)if len(bins[r])<slots),key=lambda r:(loads[r],len(bins[r]),r));bins[rank].append(expert);loads[rank]+=counts[expert];owner[expert]=rank
 assert all(len(b)==slots for b in bins)and sorted(e for b in bins for e in b)==list(range(len(counts)))
 return dict(owner_rank_by_logical_expert=owner,logical_experts_by_rank=bins,receiver_rows_by_rank=loads,maximum=max(loads))
cohorts=[]
for key,items in sorted(grouped.items()):
 rs=sorted((r for r,_ in items),key=lambda r:r['ep_rank']);phases={p for _,p in items};assert len(phases)==1
 assert len(rs)==16 and[r['ep_rank']for r in rs]==list(range(16))
 K={r['K']for r in rs};E={r['E_local']for r in rs};assert len(K)==len(E)==1;K=next(iter(K));E=next(iter(E))
 counts=[v for r in rs for v in r['receiver_expert_counts']];assert len(counts)==16*E
 receiver=[sum(r['receiver_expert_counts'])for r in rs];valid=[r['valid_rows']for r in rs];total=sum(counts);assert total==K*sum(valid)
 candidate=greedy(counts,16,E);lower=max(math.ceil(total/16),max(counts));assert candidate['maximum']>=lower
 cohorts.append(dict(role=key[0],layer=key[1],ordinal=key[2],phase=next(iter(phases)),M_by_rank=[r['M']for r in rs],valid_sender_rows_by_rank=valid,K=K,E_local=E,E_global=16*E,receiver_expert_rows_by_logical_id=counts,receiver_rows_by_rank=receiver,total_assignments=total,active_experts=sum(x>0 for x in counts),zero_receiver_ranks=sum(x==0 for x in receiver),zero_sender_positive_receiver_ranks=[r for r in range(16)if valid[r]==0 and receiver[r]>0],positive_sender_zero_receiver_ranks=[r for r in range(16)if valid[r]>0 and receiver[r]==0],maximum_expert_rows=max(counts),current_maximum_rank_rows=max(receiver),independent_unreplicated_count_lower_bound=lower,greedy_per_cohort_assignment=candidate,baseline_max_over_feasible_greedy_max=max(receiver)/candidate['maximum']if candidate['maximum']else None))
assert len(cohorts)==296
heldout=[];formal=defaultdict(list)
for c in cohorts:
 if c['phase']=='formal_C8_measure':formal[(c['role'],c['layer'])].append(c)
for (role,layer),cs in sorted(formal.items()):
 cs=sorted(cs,key=lambda c:c['ordinal']);training=cs[0];owner=training['greedy_per_cohort_assignment']['owner_rank_by_logical_expert']
 for c in cs[1:]:
  loads=[0]*16
  for expert,rows in enumerate(c['receiver_expert_rows_by_logical_id']):loads[owner[expert]]+=rows
  assert sum(loads)==c['total_assignments'];maximum=max(loads)
  heldout.append(dict(role=role,layer=layer,training_ordinal=training['ordinal'],evaluation_ordinal=c['ordinal'],phase=c['phase'],training_M_by_rank=training['M_by_rank'],evaluation_M_by_rank=c['M_by_rank'],current_maximum_rank_rows=c['current_maximum_rank_rows'],fixed_mapping_maximum_rank_rows=maximum,current_over_counterfactual_max=c['current_maximum_rank_rows']/maximum if maximum else None,counterfactual_rows_by_rank=loads,not_runtime_validation=True))
groups=defaultdict(list)
for c in cohorts:groups[(c['role'],c['phase'])].append(c)
summaries=[]
for (role,ph),cs in sorted(groups.items()):
 zero_sender=[(c,r)for c in cs for r in c['zero_sender_positive_receiver_ranks']]
 ratios=[c['baseline_max_over_feasible_greedy_max']for c in cs if c['baseline_max_over_feasible_greedy_max']is not None]
 summ=dict(role=role,phase=ph,cohorts=len(cs),active_global_experts=stats([c['active_experts']for c in cs]),current_max_rank_rows=stats([c['current_maximum_rank_rows']for c in cs]),max_single_expert_rows=stats([c['maximum_expert_rows']for c in cs]),independent_count_lower_bound=stats([c['independent_unreplicated_count_lower_bound']for c in cs]),feasible_greedy_max_rank_rows=stats([c['greedy_per_cohort_assignment']['maximum']for c in cs]),baseline_max_over_per_cohort_greedy_max=stats(ratios),zero_sender_positive_receiver_observations=len(zero_sender),zero_sender_positive_receiver_rank_counts=dict(Counter(r for c,r in zero_sender)),zero_sender_positive_receiver_rows=stats([c['receiver_rows_by_rank'][r]for c,r in zero_sender]),positive_sender_zero_receiver_observations=sum(len(c['positive_sender_zero_receiver_ranks'])for c in cs))
 summaries.append(summ)
heldsummary=[]
for role in sorted({r['role']for r in heldout}):
 rs=[r for r in heldout if r['role']==role];heldsummary.append(dict(role=role,training_rule='Per-layer first formal ordinal only; fixed owner mapping evaluated on later saved formal count samples.',evaluations=len(rs),improved_count_max=sum(r['fixed_mapping_maximum_rank_rows']<r['current_maximum_rank_rows']for r in rs),same_count_max=sum(r['fixed_mapping_maximum_rank_rows']==r['current_maximum_rank_rows']for r in rs),worse_count_max=sum(r['fixed_mapping_maximum_rank_rows']>r['current_maximum_rank_rows']for r in rs),current_over_fixed_mapping_max=stats([r['current_over_counterfactual_max']for r in rs if r['current_over_counterfactual_max']is not None]),no_retraining_on_evaluation_samples=True,no_elapsed_time_estimate=True))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_run=str(R.parent),source_snapshots=4736,complete_EP16_cohorts=len(cohorts),conservation_checks=sum(c['total_assignments']==c['K']*sum(c['valid_sender_rows_by_rank'])for c in cohorts),groups=summaries,heldout_counterfactual=heldsummary,no_new_device_model_or_source_work=True,no_join_to_other_run_timing=True,overall_complete=False,limits=['Routing/count observations and the hostprof166 native task timings are different runs; no same-call timing or route stability join.','Captured replay phase-unknown samples remain phase-unknown,not attested formal decode.','Greedy permutation is a count-only feasible counterfactual with fixed E slots per rank and unique logical-expert membership,not an implemented ABI,available EPLB mapping or hardware speedup.','Count lower bounds assume independent unreplicated assignments on16 equal row-count bins; replication,splitting,weight movement,numeric combination and link topology are separate algorithms/constraints.','Fixed first-formal mapping is evaluated only on later saved count samples; it is not end-to-end runtime validation or generalization proof.','Zero sender rows do not remove a rank that receives expert work or participates in the native protocol. Empty expert observations do not prove baseline kernels waste work on empty groups.'])
payload=(json.dumps(dict(summary=summary,record_field_schema={k:dict(type=type(v).__name__,length=len(v)if isinstance(v,(list,dict))else None)for k,v in records[0].items()},cohorts=cohorts,fixed_mapping_evaluations=heldout),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,record_field_schema={k:dict(type=type(v).__name__,length=len(v)if isinstance(v,(list,dict))else None)for k,v in records[0].items()},full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
