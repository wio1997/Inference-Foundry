#!/usr/bin/env python3
"""Intersect selected kernel-export intervals in six pre-existing rank-local windows."""
import csv,hashlib,json
from decimal import Decimal
from pathlib import Path
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
R=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
ranks={'0':'dp0_pp0_tp0_dcp0_ep0_rank0_728_20261028020238584_ascend_pt','8':'dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt'}
hashes={'0':'b42e400febf8e8b34f1dcea062f002fb2504ed9b818ea4b67d6a08e30544a8a1','8':'488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
gaps=T/'inter-forward-gap-analysis-v2/rank_gaps.csv'
assert sha(gaps)=='6c19695c6dfb5cba0bd58b23e21113f1db82f93a20a949715658ea1b81a1c917'
with gaps.open(newline='') as f: ws=[z for z in csv.DictReader(f) if z['rank'] in ranks.values() and z['target_forward_ordinal'] in ('1','6','7')]
assert len(ws)==6 and len({(z['rank'],z['target_forward_ordinal']) for z in ws})==6
def union(intervals,lo,hi):
 out=[]
 for a,b in sorted((max(lo,a),min(hi,b)) for a,b in intervals if max(lo,a)<min(hi,b)):
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
  else:out.append((a,b))
 return sum(b-a for a,b in out)
out=[]
for dev,rank in ranks.items():
 p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';assert sha(p)==hashes[dev]
 selected=[]
 with p.open(newline='') as f:
  for z in csv.DictReader(f):
   try:s=int(Decimal(z['Start Time(us)'].strip())*1000);e=s+int(Decimal(z['Duration(us)'].strip())*1000)
   except Exception:continue
   if e<=s or z['Stream ID'].strip()=='N/A' or z['Task ID'].strip()=='N/A':continue
   assert z['Device_id']==dev
   selected.append({'type':z['Type'],'device':z['Device_id'],'model':z['Model ID'].strip(),'stream':z['Stream ID'].strip(),'task':z['Task ID'].strip(),'core':z['Accelerator Core'],'s':s,'e':e})
 windows=sorted([w for w in ws if w['rank']==rank],key=lambda w:int(w['previous_target_layer39_mc2_end_ns']))
 bounds=[(int(w['previous_target_layer39_mc2_end_ns']),int(w['next_target_layer0_start_ns'])) for w in windows]
 assert all(a<b for a,b in bounds) and all(a[1]<=b[0] for a,b in zip(bounds,bounds[1:]))
 for w,(lo,hi) in zip(windows,bounds):
  rows=[z for z in selected if z['e']>lo and z['s']<hi]
  comm=[z for z in rows if z['type'].startswith('hcom_')]
  calc=[z for z in rows if z['core'] in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','AIC','AIV') and not z['type'].startswith('hcom_')]
  pairs=[]
  for c in comm:
   for k in calc:
    v=max(0,min(c['e'],k['e'],hi)-max(c['s'],k['s'],lo))
    if v:pairs.append((v,c,k))
  pairs.sort(key=lambda z:-z[0]);cu=union([(x['s'],x['e']) for x in comm],lo,hi);ku=union([(x['s'],x['e']) for x in calc],lo,hi);bu=union([(x['s'],x['e']) for x in comm+calc],lo,hi);inter=cu+ku-bu
  assert 0<=inter<=min(cu,ku) and bool(inter)==bool(pairs)
  def clean(z):return {**{k:z[k] for k in ('type','device','model','stream','task','core')},'s_rel_ns':z['s']-lo,'e_rel_ns':z['e']-lo}
  out.append({'device':dev,'ordinal':int(w['target_forward_ordinal']),'start_ns':lo,'end_ns':hi,'window_ns':hi-lo,'comm_count':len(comm),'compute_count':len(calc),'comm_streams':sorted({z['stream'] for z in comm}),'compute_streams':sorted({z['stream'] for z in calc}),'comm_clipped':sum(z['s']<lo or z['e']>hi for z in comm),'compute_clipped':sum(z['s']<lo or z['e']>hi for z in calc),'comm_union_ns':cu,'compute_union_ns':ku,'both_union_ns':bu,'intersection_ns':inter,'pair_count':len(pairs),'largest_pairs':[{'overlap_ns':v,'comm':clean(c),'compute':clean(k)} for v,c,k in pairs[:3]]})
print(json.dumps({'scope':'EP0/EP8 each ordinal 1,6,7; same-rank previous-layer39-MC2 end to next-layer0 start; positive-duration recorded CSV rows only','gaps_sha256':sha(gaps),'kernel_sha256':hashes,'windows':out,'limits':['Intersection of exported timestamps only; calibration and completeness not established.','hcom may represent summaries; MC2 internal communication cannot be separated.','No cross-device timestamp comparison, no inference about dependency, same request, real payload or speedup.','Ordinal 1 and 6 earlier eager layer attribution not independently role-validated.']},indent=2))
