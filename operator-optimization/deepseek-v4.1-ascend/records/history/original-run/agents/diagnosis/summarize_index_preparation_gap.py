#!/usr/bin/env python3
"""Summarize a fixed set of already joined profiler task intervals, not end-to-end cost."""
import hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'agents/diagnosis/indexer-attention-event-identity.json'
raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='8377661b150c2024a62622b5855c36c057c119d32493a8996a8f49c409d2badc'
rows=json.loads(raw)['pairs'];assert len(rows)==32
pairs=[]
for row in rows:
 t=[t for t in row['middle_tasks'] if t['info']['name']=='_prepare_indexer_indices_kernel']
 assert len(t)==1 and row['duration_between_ns']>0
 duration=Decimal(t[0]['duration_ns']);gap=Decimal(row['duration_between_ns'])
 assert 0<=duration<=gap
 pairs.append({'ep':row['ep'],'layer':row['layer'],'duration_us':duration/1000,'gap_us':gap/1000,'fraction':duration/gap})
def stats(values):
 return [str(min(values)),str(statistics.median(values)),str(max(values))]
result={'input_sha256':hashlib.sha256(raw).hexdigest(),'scope_count':len(pairs),'preparation_us_min_median_max':stats([x['duration_us'] for x in pairs]),'gap_us_min_median_max':stats([x['gap_us'] for x in pairs]),'per_pair_fraction_min_median_max':stats([x['fraction'] for x in pairs]),'per_pair_fraction_gt_90_percent':sum(x['fraction']>Decimal('.9') for x in pairs),'ep_count':{str(e):sum(x['ep']==e for x in pairs) for e in sorted({x['ep'] for x in pairs})},'limitations':['Selected profiled tasks only, not independent unprofiled or end-to-end latency.','Interval identity and tensor storage binding, profiler distortion, other streams and critical path unknown.','Duration includes recorded interval, not pure compute, utilization or provable savings.']}
out=root/'agents/diagnosis/index-preparation-gap-summary.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('output_sha256',hashlib.sha256(out.read_bytes()).hexdigest())
