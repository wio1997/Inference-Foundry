from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
p=A/'EP-source-granularity-replica-count-bounds-v19.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='2af3cd3b53c05596f1415d03d3569b28e1e469ba22fe38639635c8f6ecfd6a39';d=json.loads(b)
out=A/'EP-replica-bound-review-checks-v19.json';assert not out.exists()
counts=Counter();comparisons=defaultdict(Counter);examples=[]
for r in d['cohorts']:
 for c in r['cases']:
  for a in c['replica_cases']:
   assert sum(a['source_selection_bucket_counts_up_to_cyclic_rotation'])==c['rows'];counts['conservation']+=1
   R=a['replica_count'];base=a['source_selection_bucket_counts_up_to_cyclic_rotation']
   for off in range(R):
    rotated=[0]*R
    for source,n in enumerate(c['sender_rows']):rotated[(source%8+source//8+off)%R]+=n
    assert max(rotated)==max(base);counts['key_offset_maximum_invariance']+=1
  byR={a['replica_count']:a for a in c['replica_cases']};a=byR[8]['maximum_bucket_rows'];b=byR[16]['maximum_bucket_rows'];assert b<=a
  if c['selection']=='first_formal_fixed_expert'and r['is_training_sample']:continue
  k=(r['role'],r['phase'],c['selection']);comparisons[k]['samples']+=1;comparisons[k]['equal'if a==b else'lower_with_16']+=1
  if r['phase']=='formal_C8_measure'and len(examples)<2:examples.append(dict(role=r['role'],layer=r['layer'],ordinal=r['ordinal'],selection=c['selection'],rows=c['rows'],sender_rows=c['sender_rows'],maximum_bucket_rows={str(R):v['maximum_bucket_rows']for R,v in byR.items()}))
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),validation_checks=dict(counts),R8_vs_R16=[dict(role=k[0],phase=k[1],selection=k[2],**v)for k,v in sorted(comparisons.items())],examples=examples,scope='Exact arithmetic review of conditional source-granularity count model; no policy/device/model changes or elapsed-time claim.',overall_complete=False)
payload=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
