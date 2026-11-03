#!/usr/bin/env python3
"""Reproduce bounded layer-0 operator count cross-check from prior attributed CSV."""
from pathlib import Path
import csv,hashlib,json
T=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
p=T/'captured-layer-attribution/operator_metrics.csv'; q=T/'captured-layer-attribution/module-call-metrics.json';
rows=[r for r in csv.DictReader(p.open()) if r['layer']=='0' and r['sample_scope']=='INTERIOR_RUNS_1_2_3']
assert len({r['operator_type'] for r in rows})==len(rows)
selected=['HcPre','RmsNorm','RmsNormCast','DynamicQuant','QuantBatchMatmulV3','DispatchFFNCombine','HcPost','ForeachAddList','hcom_allGather_','hcom_alltoall_','hcom_reduceScatter_']
byname={x['operator_type']:x for x in rows};calls=json.loads(q.read_text()); sample=[x for x in calls if x['layer']==0 and x['run_in_model'] in (1,2,3)]
output={'status':'LAYER_WINDOW_TYPE_COUNTS_NOT_TENSOR_BINDINGS','input_sha256':{str(x.relative_to(T)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p,q)},'sample_calls':len(sample),'rank_count':len({x['rank'] for x in sample}),'selected':{x:byname.get(x) for x in selected},'method':'filter operator_metrics by layer=0 and INTERIOR_RUNS_1_2_3; counts are task types inside attributed windows, not verified producer-consumer IDs','warnings':['Operator type counts can include multiple call sites and streams; not exact tensor dependencies.','No match in these type summaries is not proof of absence in raw trace.','QuantBatchMatmulV3 may contain distinct projections.','No duration sums represent end-to-end critical-path exposure.']}
(O/'hotspot-window-evidence.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
print('sample_calls',len(sample),'ranks',output['rank_count']);
for x in selected: print(x,byname.get(x,{}).get('calls','UNKNOWN'))
