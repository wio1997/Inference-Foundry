#!/usr/bin/env python3
"""Extract the gated-router and dispatch names from existing layer aggregates."""
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'captured-layer-attribution/operator_metrics.csv';b=p.read_bytes()
with p.open() as f:rows=list(csv.DictReader(f))
names=['MoeGatingTopKHash','DispatchFFNCombine']
out={}
for name in names:
 selected=[r for r in rows if r['sample_scope']=='INTERIOR_RUNS_1_2_3' and r['operator_type']==name]
 layers=sorted(int(r['layer']) for r in selected)
 out[name]={'layer_numbers':layers,'calls_as_recorded':sorted(set(int(r['calls']) for r in selected))}
assert all(x['layer_numbers']==list(range(40)) for x in out.values())
print(json.dumps({'input_sha256':hashlib.sha256(b).hexdigest(),'result':out,
 'limitations':['same-layer grouping and same operator name do not identify source branch or tensor; task replay and count caveats apply',
                'Hash in the operator name does not establish table lookup branch',
                'No latency gain or data traffic conclusion follows'], 'no_device_work':True},indent=2))
