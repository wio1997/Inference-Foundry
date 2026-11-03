#!/usr/bin/env python3
"""Read-only sensitivity analysis of previously saved main-stream draft candidates."""
import collections, decimal, hashlib, json
from pathlib import Path
D=decimal.Decimal
T=Path(__file__).resolve().parents[2]
p=T/'eager-draft-mainstream-attribution-v5/draft_call_metrics.json'
expected='6473445dda048c157cf7e451bf959b973c5138a05e0272172a5ea98cfef21923'
raw=p.read_bytes(); assert hashlib.sha256(raw).hexdigest()==expected
rows=json.loads(raw); assert len(rows)==288
v2=T/'eager-draft-mainstream-attribution-v2/draft_call_metrics.json'
v2raw=v2.read_bytes(); assert hashlib.sha256(v2raw).hexdigest()=='0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48'
keys=lambda x: (x['rank'], x['start_ns'], x['end_ns'])
a=[x for x in rows if x['attention_count_allstreams']==0]
b=[x for x in rows if x['attention_count_allstreams']==1]
assert len(a)==126 and len(b)==162 and len({keys(x) for x in rows})==288
strict={keys(x):x for x in json.loads(v2raw)}
assert set(strict)=={keys(x) for x in b}
for x in b:
 y=strict[keys(x)]
 for k in ('aic_mac_proxy_numerator','aic_cycles_denominator','main_stream_id','model_id'):
  assert str(x[k])==str(y[k])
def stats(s):
 n=sum((D(x['aic_mac_proxy_numerator']) for x in s),D(0));d=sum((D(x['aic_cycles_denominator']) for x in s),D(0));assert d>0
 return {'windows':len(s),'ranks':len({x['rank'] for x in s}),'numerator':str(n),'denominator':str(d),'percent':str(100*n/d),'skipped_nonpositive_cycles':sum(x['skipped_no_positive_cycles'] for x in s)}
by_rank=collections.defaultdict(lambda:collections.defaultdict(list))
for x in rows: by_rank[x['rank']][x['attention_count_allstreams']].append(x)
paired=[]
for rank,groups in sorted(by_rank.items()):
 if len(groups)==2:
  a0=stats(groups[0]);b1=stats(groups[1]);paired.append({'rank':rank,'zero':a0,'one':b1,'zero_less_than_one':D(a0['percent'])<D(b1['percent'])})
out={'input_sha256':expected,'strict_input_sha256':hashlib.sha256(v2raw).hexdigest(),'zero':stats(a),'one':stats(b),'all':stats(rows),'within_rank_pairs':paired,'zero_lower_rank_count':sum(x['zero_less_than_one'] for x in paired),'scope':'Grouped selection sensitivity only. Zero exported SparseFlashMla does not prove empty query, absence of attention, draft identity, or causal effect. No cross-rank timestamp comparisons. Counts are imported aggregate CSV values with finite precision; no full-device utilization or performance speedup.'}
print(json.dumps(out,ensure_ascii=False,indent=2))
