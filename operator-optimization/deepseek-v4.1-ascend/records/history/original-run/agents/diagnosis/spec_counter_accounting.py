#!/usr/bin/env python3
"""Reconcile observed benchmark counters without assuming per-event semantics."""
import hashlib,json
from fractions import Fraction
from pathlib import Path
T=Path(__file__).resolve().parents[2];p=T/'evidence/baseline-result.json'
x=json.loads(p.read_text());rows=[]
for r in x['rounds']:
 c=r['server_counter_deltas']
 def integer(name):
  z=c[name];assert z>=0 and float(z).is_integer();return int(z)
 q=integer('generation_tokens_total');a=integer('spec_decode_num_accepted_tokens_total');d=integer('spec_decode_num_drafts_total');k=integer('spec_decode_num_draft_tokens_total');assert d>0 and k>0
 rows.append({'round':r['phase'],'generated_tokens':q,'drafts_total':d,'draft_tokens_total':k,'accepted_draft_tokens':a,'draft_tokens_per_draft':str(Fraction(k,d)),'accepted_per_draft':str(Fraction(a,d)),'accepted_over_drafted':str(Fraction(a,k)),'naive_accepted_plus_one_per_draft':a+d,'naive_minus_generated':a+d-q})
y={'scope':'per-round cumulative service counter deltas from unprofiled C8 benchmark','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows,'limitations':['A ratio of cumulative counter deltas is not an individual event distribution.','An average of five draft tokens per draft does not prove every list has length five or at most five.','Difference from accepted + drafts = generated refutes this identity in these aggregate observations but does not identify its cause.','Counts are not device-time intervals, critical-path contribution or throughput upper bounds.']}
out=T/'agents/diagnosis/spec-counter-accounting.json';assert not out.exists();out.write_text(json.dumps(y,ensure_ascii=False,indent=2)+'\n');print(json.dumps(y,ensure_ascii=False,indent=2))
