#!/usr/bin/env python3
"""Frozen target layer counter proxies: per-DP aggregation; no device access."""
from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json

T = Path(__file__).resolve().parents[2]
paths = {
 'calls': T/'captured-layer-attribution/module-call-metrics.json',
 'engine_summary': T/'captured-layer-engine-proxies/summary.json',
 'counter_audit': T/'counter-field-audit/summary.json',
 'attribution_source': T/'tools/attribute_captured_layers.py',
 'engine_source': T/'tools/analyze_layer_engine_proxies.py',
 'self': Path(__file__).resolve(),
}
sha = lambda p: sha256(p.read_bytes()).hexdigest()
cs = [x for x in json.loads(paths['calls'].read_text()) if x['run_in_model'] in (1,2,3)]
summary = json.loads(paths['engine_summary'].read_text())
audit = json.loads(paths['counter_audit'].read_text())
assert len(cs)==1920 and len({r['rank'] for r in cs})==16
assert len({(r['rank'],r['run_in_model'],r['layer']) for r in cs})==1920
assert summary['rank_count']==16 and len(summary['layers'])==40
assert audit['status']=='PASS_NO_POSITIVE_CYCLE_ROWS_WITH_MISSING_OR_INVALID_RATIOS'
assert len(audit['ranks'])==16 and not audit['bad_examples']
for layer in summary['layers']:
 for metric in ('aiv_vec_ratio','aic_mac_ratio'):
  assert layer['metrics'][metric]['missing_with_positive_cycles']==0

def ratio(rows, numerator, denominator):
 num=sum((Decimal(r[numerator]) for r in rows),Decimal(0))
 den=sum((Decimal(r[denominator]) for r in rows),Decimal(0))
 assert den>0
 return num,den,num/den

out=[]
for dp in ('dp0','dp1','all'):
 for layer in range(40):
  rows=[r for r in cs if r['layer']==layer and (dp=='all' or r['rank'].startswith(dp+'_'))]
  expected=48 if dp=='all' else 24
  assert len(rows)==expected and len({r['rank'] for r in rows})==(16 if dp=='all' else 8)
  metrics={}
  for engine,unit in (('aiv','vec'),('aic','mac')):
   num,den,value=ratio(rows,f'{engine}_{unit}_cycle_proxy_numerator',f'{engine}_cycle_denominator')
   metrics[f'{engine}_{unit}']={'numerator':str(num),'denominator':str(den),'ratio':float(value)}
   if dp=='all':
    ref=summary['layers'][layer]['metrics'][f'{engine}_{unit}_ratio']
    assert Decimal(ref['numerator'])==num and Decimal(ref['denominator'])==den
  out.append({'dp':dp,'layer':layer,'windows':len(rows),'ranks':len({r['rank'] for r in rows}),
    'mean_window_us':sum(r['window_ns'] for r in rows)/len(rows)/1000,'metrics':metrics})

groups=defaultdict(list)
for r in cs:groups[(r['rank'],r['run_in_model'])].append(r)
assert len(groups)==48 and all(len(v)==40 for v in groups.values())
lowest=Counter()
for (rank,run),rows in groups.items():
 for engine,unit in (('aiv','vec'),('aic','mac')):
  best=min(rows,key=lambda r:Decimal(r[f'{engine}_{unit}_cycle_proxy_numerator'])/Decimal(r[f'{engine}_cycle_denominator']))
  lowest[(rank.split('_')[0],f'{engine}_{unit}',best['layer'])]+=1
print(json.dumps({
 'scope':'model 49 target layer windows, per-rank run_in_model 1/2/3; no cross-rank timestamp join',
 'input_sha256':{k:sha(p) for k,p in paths.items()},
 'pooled':out,
 'individual_lowest_counts':[{'dp':a,'metric':b,'layer':c,'groups':n} for (a,b,c),n in sorted(lowest.items())],
 'positive_cycle_missing_ratio_total':{name:sum(v['metrics'][name]['missing_with_positive_cycles'] for v in summary['layers']) for name in ('aiv_vec_ratio','aic_mac_ratio')},
 'zero_or_null_cycle_rows_total':{name:sum(v['metrics'][name]['no_positive_cycles'] for v in summary['layers']) for name in ('aiv_vec_ratio','aic_mac_ratio')},
 'limitations':['CSV field proxies weighted by each engine positive-cycle denominator, not full-NPU utilization or hardware peak efficiency', 'zero/null cycle records excluded; real zero ratio with positive cycles remains included', 'event overlap / MIX counting across rows not established; AIC and AIV ratios have distinct denominators and cannot be added', 'per-rank ordinal is not proven identical request, replay, shape, stage or workload across DP groups', 'same-rank elapsed durations only; rank means are not end-to-end time', 'source-inferred attribution excludes metadata and external Engram/draft/prefill; no full-system comparison', 'pooled counts are 24 samples per DP/layer, not independent experiments'],
 'no_device_work':True},indent=2,ensure_ascii=False))
