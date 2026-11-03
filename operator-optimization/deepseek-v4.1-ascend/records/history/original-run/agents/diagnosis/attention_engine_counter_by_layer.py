#!/usr/bin/env python3
"""Recheck exported SparseFlashMla task counter proxies in fixed target windows."""
import bisect,csv,hashlib,json,statistics,sys
from collections import defaultdict,Counter
from decimal import Decimal
from pathlib import Path
T=Path(sys.argv[1]); R=Path(sys.argv[2])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
w=T/'captured-layer-attribution/module-call-metrics.json'; m=T/'evidence/TRACE-EXPORT-SHA256.json'
assert sha(w)=='d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'
assert sha(m)=='45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'
exp={v['rank']:v['kernel_csv_sha256'] for v in json.loads(m.read_text())['exports']}; windows=defaultdict(list)
for x in json.loads(w.read_text()):
 if x['run_in_model'] in (1,2,3):windows[x['rank']].append(x)
assert set(windows)==set(exp) and len(exp)==16
selected={};actual={}
for rank in sorted(exp):
 path=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';actual[rank]=sha(path);assert actual[rank]==exp[rank]
 ws=sorted(windows[rank],key=lambda x:x['window_start_ns']);assert len(ws)==120
 assert all(a['window_end_ns']<=b['window_start_ns'] for a,b in zip(ws,ws[1:]))
 starts=[x['window_start_ns'] for x in ws]; streams={str(x['main_stream_id']) for x in ws};assert len(streams)==1
 with path.open(newline='') as f:
  for z in csv.DictReader(f):
   if z['Type']!='SparseFlashMla' or z['Model ID']!='49' or z['Stream ID'] not in streams:continue
   start=Decimal(z['Start Time(us)'].strip())*1000;dur=Decimal(z['Duration(us)'])*1000
   assert start==int(start) and dur==int(dur) and dur>0
   i=bisect.bisect_right(starts,int(start))-1
   if i<0 or int(start)+int(dur)>ws[i]['window_end_ns']:continue
   win=ws[i];key=(rank,win['run_in_model'],win['layer']);assert key not in selected
   row={'duration_us':float(dur/1000),'shape':z['Input Shapes'],'outshape':z['Output Shapes'],'aic_cycles':int(z['aic_total_cycles'] or 0),'aiv_cycles':int(z['aiv_total_cycles'] or 0)}
   for field in ('aic_mac_ratio','aic_scalar_ratio','aic_mte2_ratio','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio'):
    row[field]=float(z[field]) if z[field] else None
   selected[key]=row
assert len(selected)==1920 and all((r,w['run_in_model'],w['layer']) in selected for r,ws in windows.items() for w in ws)
def summarize(rows):
 out={'count':len(rows),'duration_median_us':statistics.median(x['duration_us'] for x in rows),'duration_min_us':min(x['duration_us'] for x in rows),'duration_max_us':max(x['duration_us'] for x in rows)}
 for field in ('aic_mac_ratio','aic_scalar_ratio','aic_mte2_ratio','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio'):
  cycle='aic_cycles' if field.startswith('aic_') else 'aiv_cycles';valid=[x for x in rows if x[cycle]>0 and x[field] is not None]
  out[field+'_weighted']=sum(x[cycle]*x[field] for x in valid)/sum(x[cycle] for x in valid) if valid else None
  out[field+'_valid_count']=len(valid)
 return out
groups=[]
for run in (1,2,3):
 for layer in range(40):
  rows=[x for (r,ru,l),x in selected.items() if ru==run and l==layer]
  assert len(rows)==16
  groups.append({'run':run,'layer':layer,**summarize(rows)})
shapes=Counter((x['shape'],x['outshape']) for x in selected.values())
print(json.dumps({'scope':'16 ranks x 3 interior runs x 40 target layer windows; model 49 main stream; fully contained SparseFlashMla only','input_sha256':{'windows':sha(w),'manifest':sha(m),'kernel_by_rank':actual},'selected_count':len(selected),'overall':summarize(list(selected.values())),'groups':groups,'shape_counts':[{'input':a,'output':b,'count':n} for (a,b),n in shapes.most_common()],'limitations':['Distinct activity ratios and total cycles as exported, not whole-device utilization or exclusive time; ratios may overlap and must not be summed.','Exported per-task counts may omit tasks/engines and cannot establish workload, causal waiting, bandwidth, data identity, runtime numerical path or end-to-end speedup.','Layer groups mix rank/run and model details; comparisons do not establish matching inputs or requests.','No cross-device timestamp comparison; no new device work.']},ensure_ascii=False,indent=2))
