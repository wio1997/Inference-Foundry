#!/usr/bin/env python3
"""Verify rank0 P device CSV/trace candidate counts, not CPU/request links."""
import hashlib,json
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/p_trace_049'
SOURCES={'exact.json':'14ed159b54f3180470adcb0cc890ed0a203e8cee950a3dc71652fbdd99720a78','task_model.json':'ec2071925278f61b579dea92c2cbc3965aa777f2a53dfe371e068db437fccd72'}


def need(c,m):
 if not c:raise ValueError(m)


def load(name):
 b=(ROOT/name).read_bytes();need(hashlib.sha256(b).hexdigest()==SOURCES[name],'hash mismatch');return json.loads(b)


def verify():
 exact=load('exact.json');ids=load('task_model.json');a={(x['framework'],x['rank']):x for x in exact['ranks']};b={(x['framework'],x['rank']):x for x in ids['ranks']};need(set(a)==set(b)=={('sglang',0),('vllm',0)},'rank coverage')
 out={}
 for fw in ('sglang','vllm'):
  x=a[(fw,0)];y=b[(fw,0)];total=120 if fw=='sglang' else 399;exact_expected=35 if fw=='sglang' else 236
  need(x['source_csv_sha256']==y['source_csv_sha256'] and x['trace_sha256']==y['trace_sha256'] and x['csv_target_count']==y['csv_target_count']==total,'source identity')
  z=y['candidate_counts'];v=x['candidate_counts'];labels=('at_or_below_100us','above_100us')
  def count(d,k):return sum(d[label].get(k,0) for label in labels)
  need(count(z,'csv_events')==total and count(z,'same_stream_start_one')==total and count(z,'unique_in_both_sources_stream_start_task_model')==total,'unique candidates')
  need(count(v,'unique_in_both_sources_exact_stream_start_duration')==exact_expected and count(z,'unique_duration_exact')==exact_expected,'duration exact candidates')
  need(count(z,'unique_duration_absdiff_le_0_001us')==total,'duration difference')
  need(all(Decimal(z[label].get('unique_duration_absdiff_max_us','0'))<=Decimal('0.0005') for label in labels),'maximum duration difference')
  out[fw]={'csv_selected':total,'unique_exact_stream_start_task_model':total,'exact_duration_matches':exact_expected,'max_abs_duration_difference_us':'0.0005'}
 return {'rank0':out,'scope':'same-capture NPU0 device-event candidates only; no CPU/layer/forward/request mapping'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
