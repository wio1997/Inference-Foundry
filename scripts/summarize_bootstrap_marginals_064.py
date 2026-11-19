#!/usr/bin/env python3
"""Compare P/D bootstrap marginal histogram buckets, never request pairings."""
import argparse,hashlib,json,math
from decimal import Decimal,ROUND_CEILING,ROUND_FLOOR,localcontext
from pathlib import Path
from summarize_hist_bounds_061 import calculate
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'evidence/stage_snapshots_056/summary_v7.json'
SRC_SHA='ffb8f03c31b6d1608ec021a99d15cdf591a70880d70049b8a76f461f04774378'
OUT=ROOT/'evidence/bootstrap_marginals_064/summary_v2.json'


def need(v,msg):
 if not v:raise ValueError(msg)


def buckets(h):
 count=h['delta_count'];need(type(count) is int and count>0,'invalid count');result=[];prev=0;last=None
 for i,x in enumerate(h['buckets']):
  high=x['upper_s'];c=x['cumulative_count'];need(type(c) is int and prev<=c<=count,'bad cumulative')
  if high is None:need(i==len(h['buckets'])-1,'invalid infinity')
  else:need(type(high) in (int,float) and math.isfinite(high) and (last is None or high>last),'invalid boundaries')
  result.append((high,c-prev));last=high;prev=c
 need(prev==count and result[-1][0] is None,'invalid +Inf count');return result


def independent_slow_mean_difference(p,d,threshold):
 def as_decimal(h):
  return {'delta_count':h['delta_count'],'delta_sum_s':Decimal(str(h['delta_sum_s'])),'buckets':[{'upper_s':None if b['upper_s'] is None else Decimal(str(b['upper_s'])),'cumulative_count':b['cumulative_count']} for b in h['buckets']]}
 with localcontext() as ctx:
  ctx.prec=80
  a=calculate(as_decimal(p),Decimal(str(threshold)));b=calculate(as_decimal(d),Decimal(str(threshold)))
  if not a['slow_count'] or not b['slow_count'] or a['slow_sum_upper_s'] is None or b['slow_sum_upper_s'] is None:return None
  low=(Decimal(b['slow_sum_lower_s'])/b['slow_count']-Decimal(a['slow_sum_upper_s'])/a['slow_count'])*1000
  high=(Decimal(b['slow_sum_upper_s'])/b['slow_count']-Decimal(a['slow_sum_lower_s'])/a['slow_count'])*1000
  return {'lower_ms_outward':str(low.quantize(Decimal('0.001'),rounding=ROUND_FLOOR)),'upper_ms_outward':str(high.quantize(Decimal('0.001'),rounding=ROUND_CEILING)),'scope':'difference of independent marginal slow-event means; not per-request difference or causal attribution'}


def compare(p,d):
 a,b=buckets(p),buckets(d);need(len(a)==len(b) and all(x[0]==y[0] for x,y in zip(a,b)),'histogram boundary mismatch')
 split=[i for i,x in enumerate(a) if x[0] is not None and 0.029<x[0]<0.030];need(len(split)==1,'split threshold missing');i=split[0];out=[];low=None
 for (upper,pn),(_,dn) in zip(a,b):
  if pn or dn:out.append({'lower_exclusive_s':low,'upper_inclusive_s':upper,'prefill_count':pn,'decode_count':dn})
  low=upper
 pslow=sum(x[1] for x in a[i+1:]);dslow=sum(x[1] for x in b[i+1:]);same=all(x[1]==y[1] for x,y in zip(a[i+1:],b[i+1:]));pc=p['delta_count'];dc=d['delta_count'];match=sum(min(x[1],y[1]) for x,y in zip(a,b));different=pc-match if pc==dc else None
 return {'p_count':pc,'d_count':dc,'p_mean_s':p['mean_s'],'d_mean_s':d['mean_s'],'threshold_s':a[i][0],'p_above_threshold':pslow,'d_above_threshold':dslow,'each_above_threshold_bin_count_equal':same,'same_size_hypothetical_bijection_min_different_bin_pairs':different,'independent_slow_mean_difference':independent_slow_mean_difference(p,d,a[i][0]),'nonempty_bins':out}


def summarize():
 raw=SRC.read_bytes();need(hashlib.sha256(raw).hexdigest()==SRC_SHA,'summary SHA');doc=json.loads(raw);rows=[]
 for n in (32768,65536):
  for ph in ('cold','hot'):
   def select(role,stage):
    s=[x for x in doc['scopes'] if x.get('framework')=='sglang' and x.get('capture')=='formal' and x.get('input_tokens')==n and x.get('phase')==ph and x.get('role')==role];need(len(s)==1,'ambiguous scope');v=[x for x in s[0]['histograms'] if x['stage']==stage];need(len(v)==1,'ambiguous histogram');return v[0]
   result=compare(select('prefill','prefill_bootstrap'),select('decode','decode_bootstrap'));result.update(input_tokens=n,phase=ph);rows.append(result)
 need(all(x['each_above_threshold_bin_count_equal'] for x in rows),'slow bin equality changed');need([x['same_size_hypothetical_bijection_min_different_bin_pairs'] for x in rows]==[10,13,8,None],'bin difference changed')
 return {'schema':'sg-bootstrap-marginal-buckets-v1','source_sha256':SRC_SHA,'results':rows,'limits':['P and D histogram deltas are independently collected; no request identities or cross-role pairing','All slow-bin counts agreeing does not establish equal durations, same events, dependence or causation; broad buckets permit large within-bucket differences','P/D windows are not atomic; SG process-start/created are unavailable and hidden resets are not excluded','Minimum differently-binned pairs is only a mathematical bound under a hypothetical arbitrary bijection when counts are equal, not a claim those pairs existed','No cross-framework stage subtraction or TTFT contribution follows']}


def main():
 p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args();doc=summarize();txt=json.dumps(doc,ensure_ascii=False,indent=2)+'\n'
 if a.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(txt)
 if a.check:need(OUT.read_text(encoding='utf8')==txt,'result mismatch')
 print(json.dumps({'status':'pass','scopes':len(doc['results']),'sha256':hashlib.sha256(txt.encode()).hexdigest(),'wrote':a.write,'checked':a.check}))


if __name__=='__main__':
 try:main()
 except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):raise SystemExit('FAILED_CLOSED: source or histogram invalid')
