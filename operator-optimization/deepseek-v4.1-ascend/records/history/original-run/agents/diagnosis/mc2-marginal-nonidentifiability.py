#!/usr/bin/env python3
"""Illustrate, conditionally, why two marginal ratios do not identify overlap/wait.
Only reads an existing aggregated task record; no device access.
"""
import hashlib,json
from pathlib import Path
from decimal import Decimal
p=Path('agents/diagnosis/layer0-engine-work-proxies.json')
raw=p.read_bytes(); d=json.loads(raw, parse_float=Decimal)
rows=[r for r in d['per_rank'] if '_ep8_' in r['rank'] and r['run']==1]
assert len(rows)==1
r=rows[0]; a=r['mac_ratio']; b=r['mte2_ratio']
assert a==Decimal('0.01') and b==Decimal('0.078')
# Integer activity units are a constructive counterexample, not hardware cycles.
n=1000; A=set(range(10)); B_disjoint=set(range(10,88)); B_nested=set(range(78))
assert len(A)/n==float(a) and len(B_disjoint)/n==float(b) and len(B_nested)/n==float(b)
out={'source_sha256':hashlib.sha256(raw).hexdigest(),'sample':{'rank':'EP8','run':1,'duration_us':str(r['duration_us']),'mac_ratio':str(a),'mte2_ratio':str(b)},'assumption':'Each ratio is an exact measure of an activity set on the same normalized core-cycle sample space; this has not been established for the historical exporter.', 'bounds': {'both_min':str(max(Decimal(0),a+b-1)), 'both_max':str(min(a,b)), 'neither_min':str(max(Decimal(0),1-a-b)), 'neither_max':str(1-max(a,b))},'witnesses':[]}
for name,B in [('disjoint',B_disjoint),('nested',B_nested)]:
    out['witnesses'].append({'name':name,'units':n,'mac':len(A),'mte2':len(B),'both':len(A&B),'neither':n-len(A|B)})
print(json.dumps(out,ensure_ascii=False,indent=2))
