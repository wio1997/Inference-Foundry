#!/usr/bin/env python3
import argparse,csv,hashlib,json,statistics
from collections import defaultdict,Counter
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args();T=Path(a.task_root);R=Path(a.trace_root)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p,s):
 p=T/p
 if sha(p)!=s:raise ValueError('input hash mismatch '+str(p))
 return json.loads(p.read_text())
ws=read('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a');manifest=read('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139');summary=read('agents/diagnosis/matmul-hcpre-shapes.json','2d09b0226e8882039d9b866848ab3a9b7625b66f67787b3eb44cf81d0439028e');prev=read('agents/fusion/q-scatter-16rank-overlap.json','4d58d6c00eadb7c146670be2ea49157a1beb666d5745e7fe06859c4ce9c8fb7a')
g=[x for x in summary['groups'] if x['type']=='QuantBatchMatmulV3' and x['signature']['Output Shapes']=='"3,32768"'];assert len(g)==1 and g[0]['count']==1920;sig=g[0]['signature'];expected={e['rank']:e['kernel_csv_sha256'] for e in manifest['exports']};windows=defaultdict(list);old={(x['rank'],x['run'],x['layer']):x for x in prev['samples']};assert len(old)==1920
for w in ws:
 if w['run_in_model'] in (1,2,3):windows[w['rank']].append(w)
if len(expected)!=16 or set(expected)!=set(windows) or any(len(v)!=120 for v in windows.values()):raise ValueError('unexpected windows')
rows=[];hashes={}
for rank in sorted(expected):
 p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';hashes[rank]=sha(p)
 if hashes[rank]!=expected[rank]:raise ValueError('csv hash mismatch')
 w=windows[rank];ev=defaultdict(list)
 if any(str(q['model_id'])!='49' for q in w) or len({str(q['main_stream_id']) for q in w})!=1:raise ValueError('model/stream mismatch')
 with p.open(newline='') as f:
  for z in csv.DictReader(f):
   if z['Model ID']!='49' or z['Type'] not in ('QuantBatchMatmulV3','ScatterNdUpdateSk','InplacePartialRotaryMul','SparseFlashMla'):continue
   start=Decimal(z['Start Time(us)'].strip())*1000;dur=Decimal(z['Duration(us)'].strip())*1000
   if not(start.is_finite() and dur.is_finite() and start==start.to_integral_value() and dur==dur.to_integral_value() and dur>0):continue
   start=int(start);end=start+int(dur);hits=[x for x in w if x['window_start_ns']<=start and end<=x['window_end_ns']]
   if len(hits)>1:raise ValueError('overlapping windows')
   if not hits:continue
   x=hits[0];main=str(x['main_stream_id']);key=(x['run_in_model'],x['layer']);stream=z['Stream ID'];typ=z['Type'];shape=z['Input Shapes'].strip('"')
   isq=typ=='QuantBatchMatmulV3' and stream==main and all(z[k]==v for k,v in sig.items() if k!='kernel_name_sha256') and hashlib.sha256(z['Name'].encode()).hexdigest()==sig['kernel_name_sha256']
   iss=typ=='ScatterNdUpdateSk' and stream!=main and stream.isdigit() and shape.endswith(';24,2;24,512')
   isr=typ=='InplacePartialRotaryMul' and stream==main and shape.startswith('3,1,64,512;')
   isa=typ=='SparseFlashMla' and stream==main
   if isq or iss or isr or isa:ev[key].append({'kind':'q' if isq else 'scatter' if iss else 'rope' if isr else 'attn','stream':stream,'task':z['Task ID'],'start_ns':start,'end_ns':end})
 for x in w:
  v=ev[(x['run_in_model'],x['layer'])];q=[y for y in v if y['kind']=='q'];s=[y for y in v if y['kind']=='scatter'];at=[y for y in v if y['kind']=='attn'];rr=[y for y in v if y['kind']=='rope'];key=(rank,x['run_in_model'],x['layer']);prior=old[key]
  if len(q)!=1 or len(s)!=1:raise ValueError('Q or scatter no longer unique')
  q=q[0];s=s[0];over=max(0,min(q['end_ns'],s['end_ns'])-max(q['start_ns'],s['start_ns']))
  if (q['task'],s['task'],q['end_ns']-q['start_ns'],over,s['end_ns']-q['end_ns'])!=(prior['q_task'],prior['scatter_task'],prior['q_duration_ns'],prior['overlap_ns'],prior['scatter_tail_after_q_ns']):raise ValueError('prior mismatch')
  pre=[r for r in rr if len(at)==1 and r['end_ns']<=at[0]['start_ns']];r={'rank':rank,'run':x['run_in_model'],'layer':x['layer'],'attn_count':len(at),'all_main_rope_count':len(rr),'pre_attention_rope_count':len(pre)}
  if len(at)==1 and len(pre)==1:
   r['rope_task']=pre[0]['task'];r['rope_duration_ns']=pre[0]['end_ns']-pre[0]['start_ns'];r['scatter_end_to_rope_start_ns']=pre[0]['start_ns']-s['end_ns'];r['q_end_to_rope_start_ns']=pre[0]['start_ns']-q['end_ns'];r['rope_end_to_attention_start_ns']=at[0]['start_ns']-pre[0]['end_ns']
  rows.append(r)
if len(rows)!=1920:raise ValueError('incorrect total')
def summ(v):
 ok=[x for x in v if x['attn_count']==1 and x['pre_attention_rope_count']==1];d=[x['scatter_end_to_rope_start_ns'] for x in ok]
 return {'windows':len(v),'attention_count_hist':dict(sorted(Counter(str(x['attn_count']) for x in v).items())),'all_rope_count_hist':dict(sorted(Counter(str(x['all_main_rope_count']) for x in v).items())),'pre_rope_count_hist':dict(sorted(Counter(str(x['pre_attention_rope_count']) for x in v).items())),'selected_unique':len(ok),'gap_positive':sum(t>0 for t in d),'gap_zero':sum(t==0 for t in d),'gap_negative':sum(t<0 for t in d),'scatter_end_to_rope_start_ns_min_median_max':[min(d),statistics.median(d),max(d)] if d else None,'rope_duration_ns_median':statistics.median([x['rope_duration_ns'] for x in ok]) if d else None}
out={'scope':'16 rank x 3 interior runs x 40 windows; each selected Q/scatter bound to prior same-rank per-window results; main stream and model49; pre-attention shape candidate unique before sole SMLA','input_csv_sha256':hashes,'samples':rows,'overall':summ(rows),'by_rank':{rank:summ([x for x in rows if x['rank']==rank]) for rank in sorted(expected)},'by_run':{str(run):summ([x for x in rows if x['run']==run]) for run in (1,2,3)},'limitations':['Candidate shape and before-attention phase not tensor identity or Python role.','Same-rank cross-stream exported clock requires calibration; no event object relationship or waiting inferred.','Incomplete export and profiler perturbation, and 3 internal runs from same profile; no independent repetitions or same workload claim.','No cross-rank clock subtraction; no conclusion of causality, idle time, critical path, or speedup.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
