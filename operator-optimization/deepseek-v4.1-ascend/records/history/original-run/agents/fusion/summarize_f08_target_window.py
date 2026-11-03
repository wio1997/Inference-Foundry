#!/usr/bin/env python3
"""Summarize same-name task rows in existing captured-layer CSV; no device access."""
import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'captured-layer-attribution/operator_metrics.csv';b=p.read_bytes()
names=['ScatterNdUpdateSk','InplacePartialRotaryMul','RmsNorm','SparseFlashMla']
with p.open() as f:rows=list(csv.DictReader(f))
out={}
for name in names:
 selected=[r for r in rows if r['sample_scope']=='INTERIOR_RUNS_1_2_3' and r['operator_type']==name]
 layers=sorted(int(r['layer']) for r in selected)
 out[name]={'layer_numbers':layers,'unique_layers':len(set(layers)),
            'calls_distinct_as_recorded':sorted(set(int(r['calls']) for r in selected))}
assert all(x['layer_numbers']==list(range(40)) for x in out.values())
print(json.dumps({'input_sha256':hashlib.sha256(b).hexdigest(),'scope':'INTERIOR_RUNS_1_2_3',
                  'result':out,'limitations':['Operator name and layer attribution do not establish callsite or tensor identity.',
                  'Calls aggregated in windows are not necessarily unique independent requests or kernel invocations.',
                  'No direct proof of F08 internal connection or saved time.'],'no_device_work':True},indent=2))
