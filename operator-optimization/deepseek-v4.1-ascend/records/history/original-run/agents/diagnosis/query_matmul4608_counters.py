#!/usr/bin/env python3
import argparse, csv, hashlib, json, statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--task-root',required=True);p.add_argument('--trace-root',required=True);a=p.parse_args();t=Path(a.task_root);trace=Path(a.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(rel,want):
 p=t/rel
 if sha(p)!=want:raise ValueError('input hash mismatch '+rel)
 return json.loads(p.read_text())
wins=read('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a')
man=read('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139')
shapes=read('agents/diagnosis/matmul-hcpre-shapes.json','2d09b0226e8882039d9b866848ab3a9b7625b66f67787b3eb44cf81d0439028e')
g=[x for x in shapes['groups'] if x['type']=='QuantBatchMatmulV3' and x['signature']['Output Shapes']=='"3,4608"']
if len(g)!=1 or g[0]['count']!=1920:raise ValueError('target signature mismatch')
sig=g[0]['signature']; expected={x['rank']:x['kernel_csv_sha256'] for x in man['exports']}; windows=defaultdict(list)
for w in wins:
 if w['run_in_model'] in (1,2,3):windows[w['rank']].append(w)
if len(expected)!=16 or set(expected)!=set(windows) or any(len(v)!=120 for v in windows.values()):raise ValueError('bad window dims')
rows=[];counts=defaultdict(int); actual={}; metrics=['aic_mac_ratio','aic_mte2_ratio','aic_scalar_ratio']
for rank in sorted(expected):
 path=trace/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';actual[rank]=sha(path)
 if actual[rank]!=expected[rank]:raise ValueError('CSV hash mismatch '+rank)
 ws=windows[rank];streams={str(w['main_stream_id']) for w in ws}
 if any(str(w['model_id'])!='49' for w in ws) or len(streams)!=1:raise ValueError('unexpected model/streams')
 with path.open(newline='') as f:
  reader=csv.DictReader(f); needed=set(list(sig.keys())+['Name','Type','Model ID','Stream ID','Start Time(us)','Duration(us)','aic_total_cycles']+metrics)-{'kernel_name_sha256'}
  if not needed.issubset(reader.fieldnames):raise ValueError('missing CSV columns '+str(needed-set(reader.fieldnames)))
  for z in reader:
   if z['Type'].strip()!='QuantBatchMatmulV3' or z['Model ID']!='49' or z['Stream ID'] not in streams:continue
   if any(z[k]!=v for k,v in sig.items() if k!='kernel_name_sha256') or hashlib.sha256(z['Name'].encode()).hexdigest()!=sig['kernel_name_sha256']:continue
   try:
    start=Decimal(z['Start Time(us)'])*1000;dur=Decimal(z['Duration(us)'])*1000
    if not(start.is_finite() and dur.is_finite() and dur>0 and start==start.to_integral_value() and dur==dur.to_integral_value()):raise ValueError('bad time')
   except (InvalidOperation,ValueError):raise ValueError('invalid target time')
   start=int(start);dur=int(dur);hits=[w for w in ws if w['window_start_ns']<=start and start+dur<=w['window_end_ns']]
   if len(hits)>1:raise ValueError('overlapping windows')
   if not hits:continue
   w=hits[0]; key=(rank,w['run_in_model'],w['layer']);counts[key]+=1
   rows.append({'rank':rank,'dp':rank.split('_')[0],'run':w['run_in_model'],'layer':w['layer'],'duration_us':dur/1000,'aic_total_cycles':z['aic_total_cycles'], **{k:z[k] for k in metrics}})
for rank,ws in windows.items():
 for w in ws:
  if counts[(rank,w['run_in_model'],w['layer'])]!=1:raise ValueError('target count != 1 in window')
if len(rows)!=1920:raise ValueError('bad total')
def num(s):
 try:
  x=Decimal(s)
  return x if x.is_finite() else None
 except (InvalidOperation,ValueError,TypeError):return None
def summary(data):
 d=[x['duration_us'] for x in data]; ret={'count':len(data),'duration_us':{'min':min(d),'median':statistics.median(d),'max':max(d)},'metrics':{}}
 for metric in metrics:
  chosen=[]; badcycle=0; badratio=0
  for x in data:
   cyc=num(x['aic_total_cycles']);val=num(x[metric])
   if cyc is None or cyc<=0:badcycle+=1
   elif val is None or val<0 or val>1:badratio+=1
   else:chosen.append((cyc,val))
  c=sum((x[0] for x in chosen),Decimal(0));v=[float(x[1]) for x in chosen]
  ret['metrics'][metric]={'included':len(v),'bad_cycle':badcycle,'bad_ratio_given_positive_cycle':badratio,'min':min(v) if v else None,'median':statistics.median(v) if v else None,'max':max(v) if v else None,'cycle_weighted_mean':float(sum((a*b for a,b in chosen),Decimal(0))/c) if c else None,'cycles_sum':str(c)}
 return ret
groups=defaultdict(list)
for x in rows:
 groups[('dp',x['dp'])].append(x);groups[('rank',x['rank'])].append(x);groups[('dp_run',x['dp']+':'+str(x['run']))].append(x)
out={'scope':'16 ranks x 3 runs x 40 windows, same model49 main stream, same exact physical signature; complete containment','signature':sig,'csv_sha256_by_rank':actual,'overall':summary(rows),'grouped':{typ:{key:summary(v) for (tp,key),v in sorted(groups.items()) if tp==typ} for typ in ('dp','rank','dp_run')},'limitations':['Ratios are reported cycle-based task metrics, not whole-chip utilisation, measured FLOPs or HBM bytes.','Same signature does not prove same work, tensor identity, effective rows, loaded binary or common request.','Same captured profile runs are not independent benchmark runs; rank and stream times cannot be summed into wall time.','Export version, field precision, completeness, and profiling perturbation remain unresolved.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
