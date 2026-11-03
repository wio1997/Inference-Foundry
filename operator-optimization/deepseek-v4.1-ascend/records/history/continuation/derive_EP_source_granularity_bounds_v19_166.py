from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,gzip,socket,statistics,math
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
OUT=A/'profile-readonly-20261031/EP-source-granularity-replica-count-bounds-v19.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
d=pin(A/'profile-readonly-20261031/EP-row-distribution-and-placement-counterfactual-v17.json','4dd1ffc5aaf63d7ad54bcdd23f1a9435962a319976d7c9df8ecd7b5bde1b1c69')
builder=A/'hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15/EP-placement-map-builder-callees-v19.json'
pin(builder,'54e713a89594e810e95cd3f09bd5671c048949b62c687bc537a0bf826de7f829')
p=A/'observer-20261030T160315Z-76b79f0f/value-review/decoded-records.jsonl.gz';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='d6e787e5bd8498c8e68ccd3f2d1874a8dfd20260071fae7f16ad6876ec3c3029'
groups=defaultdict(list)
for line in gzip.decompress(b).decode().splitlines():
 r=json.loads(line);groups[(r['role'],r['layer'],r['ordinal'])].append(r)
training={}
for c in d['cohorts']:
 if c['phase']=='formal_C8_measure':
  k=c['role'],c['layer']
  if k not in training or c['ordinal']<training[k]['ordinal']:training[k]=c
rows=[]
for c in d['cohorts']:
 rs=sorted(groups[(c['role'],c['layer'],c['ordinal'])],key=lambda r:r['ep_rank']);assert [r['ep_rank']for r in rs]==list(range(16))
 ex=c['receiver_expert_rows_by_logical_id'];hot=max(range(len(ex)),key=lambda i:(ex[i],-i));selected=[('per_cohort_hottest',hot)]
 train=training.get((c['role'],c['layer']))
 if train is not None and c['phase']=='formal_C8_measure':
  tex=train['receiver_expert_rows_by_logical_id'];selected.append(('first_formal_fixed_expert',max(range(len(tex)),key=lambda i:(tex[i],-i))))
 cases=[]
 for kind,e in selected:
  hist=[int(r['sender_expert_histogram'].get(str(e),0))for r in rs];assert sum(hist)==ex[e]
  replicas=[]
  for R in [2,4,8,16]:
   # Omit key offset: adding e modulo R only cyclically rotates this bucket vector.
   buckets=[0]*R
   for rank,n in enumerate(hist):buckets[(rank%8+rank//8)%R]+=n
   assert sum(buckets)==sum(hist)
   bound=max(max(hist),math.ceil(sum(hist)/R));assert max(buckets)>=bound
   replicas.append(dict(replica_count=R,source_selection_bucket_counts_up_to_cyclic_rotation=buckets,maximum_bucket_rows=max(buckets),ideal_unrestricted_assignment_lower_bound=math.ceil(sum(hist)/R),source_atomic_lower_bound=bound,nonempty_buckets=sum(n>0 for n in buckets),reachable_buckets_from_all_EP16_senders=len({(r%8+r//8)%R for r in range(16)})))
  cases.append(dict(selection=kind,expert_native_label=e,rows=sum(hist),sender_rows=hist,active_senders=sum(n>0 for n in hist),maximum_sender_rows=max(hist),replica_cases=replicas))
 rows.append(dict(role=c['role'],phase=c['phase'],layer=c['layer'],ordinal=c['ordinal'],training_ordinal=train['ordinal']if train else None,is_training_sample=bool(train and train['ordinal']==c['ordinal']),cases=cases))
def stats(xs):return dict(n=len(xs),minimum=min(xs),median=statistics.median(xs),maximum=max(xs))if xs else None
groupsummary=[]
for role,phase in sorted({(r['role'],r['phase'])for r in rows}):
 rr=[r for r in rows if(r['role'],r['phase'])==(role,phase)];summ=[]
 for selection in ['per_cohort_hottest','first_formal_fixed_expert']:
  cc=[c for r in rr for c in r['cases']if c['selection']==selection and(selection!='first_formal_fixed_expert'or not r['is_training_sample'])]
  if not cc:continue
  ss=[]
  for R in [2,4,8,16]:
   vals=[(c,next(a for a in c['replica_cases']if a['replica_count']==R))for c in cc]
   ss.append(dict(replica_count=R,samples=len(vals),maximum_bucket_rows=stats([a['maximum_bucket_rows']for c,a in vals]),hot_rows_over_maximum_bucket=stats([c['rows']/a['maximum_bucket_rows']for c,a in vals if a['maximum_bucket_rows']]),worse_than_ideal_assignment_bound=sum(a['maximum_bucket_rows']>a['ideal_unrestricted_assignment_lower_bound']for c,a in vals),nonempty_buckets=stats([a['nonempty_buckets']for c,a in vals])))
  summ.append(dict(selection=selection,samples=len(cc),active_senders=stats([c['active_senders']for c in cc]),single_sender_samples=sum(c['active_senders']==1 for c in cc),replica_summaries=ss))
 groupsummary.append(dict(role=role,phase=phase,cohorts=len(rr),selections=summ))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),cohorts=len(rows),groups=groupsummary,formula='Existing generate_log2phy_map: replica_index=(ep_rank%TP + ep_rank//TP + logical_expert_key)%R, TP=8. Expert-key offset is a cyclic bucket rotation and cannot change max bucket rows.',all_EP16_sender_reachable_buckets={str(R):len({(r%8+r//8)%R for r in range(16)})for R in [2,4,8,16]},actual_redundancy_in_observer=0,no_new_source_DB_device_or_service_work=True,overall_complete=False,limits=['Conditional source-granularity count model using actual saved sender histograms; no replicas or mappings were applied.','R physical copies and uniform-capacity/weight-loaded placement are assumptions for this per-expert model,not a constructed whole-model feasible layout.','All assignments of one source rank to one logical expert select the same replica in this source path. It does not provide per-token splitting on that rank.','Native-label-to-logical-key mapping remains unpromoted; key offset only rotates bucket counts,so the maximum count is independent of this unknown offset.','Different experts may share receiver ranks; maximum bucket is not full receiver maximum,elapsed time,physical utilization or speedup.','Per-cohort hottest selection is an oracle diagnostic. First-formal fixed-expert results use later samples only and still do not validate an end-to-end placement policy.','Observer route counts and later hostprof native timing remain separate runs.'])
payload=(json.dumps(dict(summary=summary,cohorts=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,first_cohort=rows[0],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
