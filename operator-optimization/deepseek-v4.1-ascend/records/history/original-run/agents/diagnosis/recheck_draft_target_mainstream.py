#!/usr/bin/env python3
"""Verify saved partial main-stream target/draft cycle-proxy ranking; no device."""
import csv, hashlib, json
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

T = Path(__file__).resolve().parents[2]
files = {
 'metrics': ('eager-draft-mainstream-attribution-v2/module_metrics.csv','9e96d35b57fe4652da6e955e3a0f1318f8638b5094f42b12c9c2299c3b43fe23'),
 'calls': ('eager-draft-mainstream-attribution-v2/draft_call_metrics.json','0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48'),
 'summary': ('eager-draft-mainstream-attribution-v2/summary.json','1125b875dceb727ee5bf21c1816cbed20fcc7ce91e3b2fc30a0054a18e36e43b'),
 'diagnosis': ('evidence/draft-module-diagnosis.json','fce16d6fec122c082a7cb3abe4b5762111f944fca7f3cc1477dc93b550062b11'),
}
for name,(rel,expected) in files.items():
 if hashlib.sha256((T/rel).read_bytes()).hexdigest()!=expected: raise RuntimeError('INPUT_HASH_MISMATCH '+name)
with (T/files['metrics'][0]).open() as f: rows=list(csv.DictReader(f))
calls=json.loads((T/files['calls'][0]).read_text()); summary=json.loads((T/files['summary'][0]).read_text()); diagnosis=json.loads((T/files['diagnosis'][0]).read_text())
draft=[r for r in rows if r['kind']=='EAGER_DRAFT_POOLED']; ranks={r['rank'] for r in draft};
if len(draft)!=15 or len(ranks)!=15 or len(calls)!=162: raise RuntimeError('COVERAGE_ERROR')
byrank=defaultdict(list)
for c in calls: byrank[c['rank']].append(c)
if set(byrank)!=ranks: raise RuntimeError('DRAFT_RANK_SET_MISMATCH')
for d in draft:
 group=sorted(byrank[d['rank']],key=lambda x:x['start_ns'])
 if len(group)!=int(d['calls']) or any(a['end_ns']>b['start_ns'] for a,b in zip(group,group[1:])): raise RuntimeError('DRAFT_WINDOW_COVERAGE')
 if sum((Decimal(x['aic_mac_proxy_numerator']) for x in group),Decimal(0)) != Decimal(d['aic_mac_proxy_numerator']): raise RuntimeError('DRAFT_NUMERATOR_MISMATCH')
 if sum((Decimal(x['aic_cycles_denominator']) for x in group),Decimal(0)) != Decimal(d['aic_cycles_denominator']): raise RuntimeError('DRAFT_DENOMINATOR_MISMATCH')
 if sum(int(x['skipped_no_positive_cycles']) for x in group)!=int(d['skipped_no_positive_cycles']): raise RuntimeError('DRAFT_SKIP_MISMATCH')
target=[r for r in rows if r['kind']=='CAPTURED_TARGET' and r['rank'] in ranks]
for layer in range(40):
 rr=[r for r in target if r['module']==f'target.layers.{layer}.forward']
 if len(rr)!=15 or {x['rank'] for x in rr}!=ranks or any(int(x['calls'])!=3 for x in rr): raise RuntimeError('TARGET_COVERAGE')
def ratio(rr):
 n=sum((Decimal(x['aic_mac_proxy_numerator']) for x in rr),Decimal(0)); d=sum((Decimal(x['aic_cycles_denominator']) for x in rr),Decimal(0))
 if d<=0: raise RuntimeError('EMPTY_DENOMINATOR')
 return n,d,n/d
ranked=[]
for label,rr in [('draft.decoder_block.pooled',draft)]+[(f'target.layers.{i}.forward',[r for r in target if r['module']==f'target.layers.{i}.forward']) for i in range(40)]:
 n,d,v=ratio(rr); ranked.append({'module':label,'numerator':str(n),'denominator':str(d),'ratio':str(v),'calls':sum(int(x['calls']) for x in rr),'skipped_nonpositive_cycles':sum(int(x['skipped_no_positive_cycles']) for x in rr)})
ranked.sort(key=lambda x:Decimal(x['ratio']))
exact_group_order=sorted(ranked,key=lambda x:Fraction(Decimal(x['numerator']))/Fraction(Decimal(x['denominator'])))
if [x['module'] for x in exact_group_order]!=[x['module'] for x in ranked]: raise RuntimeError('GROUP_ORDER_PRECISION_MISMATCH')
for x,y in zip(ranked,diagnosis['strict']['ranked_groups']):
 if x['module']!=y['module'] or x['numerator']!=y['numerator'] or x['denominator']!=y['denominator']: raise RuntimeError('SUMMARY_MISMATCH')
paired=[]
def exact_ratio(row):
 n=Decimal(row['aic_mac_proxy_numerator']); d=Decimal(row['aic_cycles_denominator'])
 if d<=0: raise RuntimeError('EMPTY_PER_RANK_DENOMINATOR')
 return Fraction(n)/Fraction(d)

for d in draft:
 others=[r for r in target if r['rank']==d['rank']]
 exact_order=sorted(others,key=exact_ratio)
 csv_order=sorted(others,key=lambda x:Decimal(x['ratio']))
 if [x['module'] for x in exact_order]!=[x['module'] for x in csv_order]: raise RuntimeError('PER_RANK_TARGET_ORDER_ROUNDING_MISMATCH')
 lowest=exact_order[0]; layer0=next(x for x in others if x['module']=='target.layers.0.forward')
 draft_below_target0=exact_ratio(d)<exact_ratio(layer0)
 draft_below_own_target_min=exact_ratio(d)<exact_ratio(lowest)
 if draft_below_target0 != (Decimal(d['ratio'])<Decimal(layer0['ratio'])) or draft_below_own_target_min != (Decimal(d['ratio'])<Decimal(lowest['ratio'])): raise RuntimeError('PER_RANK_DRAFT_ORDER_ROUNDING_MISMATCH')
 paired.append({'dp':d['rank'].split('_')[0], 'draft_below_target0':draft_below_target0, 'draft_below_own_target_min':draft_below_own_target_min, 'draft_calls':int(d['calls'])})
print(json.dumps({'scope':'15 ranks, strict162 draft pooled versus 40 target layers with45 windows each; main stream only', 'sample_counts_per_rank':dict(sorted(Counter(x['calls'] for x in draft).items())), 'first_three':ranked[:3], 'draft_below_target0_per_rank':sum(x['draft_below_target0'] for x in paired), 'draft_below_own_target_min_per_rank':sum(x['draft_below_own_target_min'] for x in paired), 'per_dp':{dp:{'n':sum(x['dp']==dp for x in paired),'below_own_target_min':sum(x['dp']==dp and x['draft_below_own_target_min'] for x in paired)} for dp in ('dp0','dp1')}, 'per_rank_exact_fraction_vs_saved_ratio_order':'IDENTICAL', 'input_sha256':{k:v[1] for k,v in files.items()}, 'limits':['Sample-selected pooled draft is not an individual block or global framework ranking.', 'AIC cycle proxies are not NPU occupancy and exclude cross-stream and nonpositive cycle rows.', 'Exact arithmetic does not recover precision already lost in underlying CSV task-level ratio fields.', 'No NPU use.']},ensure_ascii=False,indent=2))
