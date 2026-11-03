#!/usr/bin/env python3
"""Read-only, predetermined 35→36,36→37,37→38,38→39 rank-local gaps; no device work."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);wp=T/'captured-layer-attribution/module-call-metrics.json';mp=T/'evidence/TRACE-EXPORT-SHA256.json'
expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']}
w=defaultdict(dict);boundaries=(35,36,37,38);needed={35,36,37,38,39}
for x in json.loads(wp.read_text()):
 if x['run_in_model'] in (1,2,3) and x['layer'] in needed:
  key=(x['rank'],x['run_in_model']);assert x['layer'] not in w[key];w[key][x['layer']]=x
assert len(w)==48 and len({k[0] for k in w})==16 and {k[0] for k in w}==set(expected)
by=defaultdict(list)
for (rank,run),pair in w.items():
 assert set(pair)==needed
 for after in boundaries:
  x,y=pair[after],pair[after+1];assert str(x['model_id'])==str(y['model_id'])=='49' and str(x['main_stream_id'])==str(y['main_stream_id'])
  assert x['window_end_ns']<y['window_start_ns']
  by[rank].append({'run':run,'after_layer':after,'start':x['window_end_ns'],'end':y['window_start_ns'],'main':str(x['main_stream_id'])})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def union(spans):
 merged=[]
 for lo,hi in sorted(spans):
  if not merged or lo>merged[-1][1]:merged.append([lo,hi])
  else:merged[-1][1]=max(merged[-1][1],hi)
 return sum(b-a for a,b in merged)
rows=[];hashes={}
for rank,gs in sorted(by.items()):
 assert len(gs)==12;gs.sort(key=lambda g:g['start']);assert all(x['end']<=y['start'] for x,y in zip(gs,gs[1:]))
 p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';hashes[rank]=sha(p);assert hashes[rank]==expected[rank]
 records={(g['run'],g['after_layer']):[] for g in gs}
 with p.open() as f:
  for z in csv.DictReader(f):
   start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
   if end<=start:continue
   for g in gs:
    if start>=g['end'] or end<=g['start']:continue
    records[(g['run'],g['after_layer'])].append({'type':z['Type'],'stream':z['Stream ID'].strip(),'model':z['Model ID'].strip(),
      'event_start_rel_ns':start-g['start'],'event_end_rel_ns':end-g['start'],
      'clipped_start_rel_ns':max(start,g['start'])-g['start'],'clipped_end_rel_ns':min(end,g['end'])-g['start'],
      'input_shapes':z['Input Shapes'],'input_dtypes':z['Input Data Types'],'output_shapes':z['Output Shapes'],'output_dtypes':z['Output Data Types']})
 for g in gs:
  ev=records[(g['run'],g['after_layer'])];assert len(ev)<=300,'unexpected event count, inspect explicitly'
  num=[x for x in ev if x['stream'].isdigit()];unknown=[x for x in ev if not x['stream'].isdigit()]
  main=[x for x in num if x['stream']==g['main'] and x['model']=='49'];other=[x for x in num if x['stream']!=g['main']];main_other_model=[x for x in num if x['stream']==g['main'] and x['model']!='49']
  spans=lambda es:[(x['clipped_start_rel_ns'],x['clipped_end_rel_ns']) for x in es]
  duration=g['end']-g['start'];nu=union(spans(num));un=union(spans(unknown));all_=union(spans(ev));assert all_<=duration
  typ=defaultdict(list)
  for x in num:typ[x['type']].append((x['clipped_start_rel_ns'],x['clipped_end_rel_ns']))
  rows.append({'rank':rank,'run':g['run'],'after_layer':g['after_layer'],'main_stream':g['main'],'gap_ns':duration,
   'numeric_union_ns':nu,'unknown_union_ns':un,'all_rows_union_ns':all_,
   'numeric_unknown_overlap_ns':nu+un-all_,'not_covered_by_any_exported_row_ns':duration-all_,
   'main_model49_union_ns':union(spans(main)),'main_other_model_union_ns':union(spans(main_other_model)),'other_numeric_stream_union_ns':union(spans(other)),
   'numeric_types':{k:{'rows':len(v),'union_ns':union(v)} for k,v in sorted(typ.items())},'events':ev})
assert len(rows)==192 and len(hashes)==16
summary=[]
for after in boundaries:
 for run in (1,2,3):
  rr=[r for r in rows if r['run']==run and r['after_layer']==after];assert len(rr)==16
  row={'run':run,'after_layer':after}
  for name in ('gap_ns','numeric_union_ns','unknown_union_ns','all_rows_union_ns','numeric_unknown_overlap_ns','not_covered_by_any_exported_row_ns','main_model49_union_ns','main_other_model_union_ns','other_numeric_stream_union_ns'):
   vals=[r[name]/1000 for r in rr];row[name.replace('_ns','_us')]={'min':min(vals),'median':statistics.median(vals),'max':max(vals)}
  row['ranks_with_matmul']=sum(any(e['type']=='MatMulV2' for e in r['events']) for r in rr)
  row['matmul_union_us_median']=statistics.median(r['numeric_types'].get('MatMulV2',{'union_ns':0})['union_ns'] for r in rr)/1000
  summary.append(row)
out={'scope':'192 rank-local intervals: ends of captured layer brackets 35,36,37,38 to the next layer; 16 ranks x 3 profile samples',
 'control_selection':'35→36 is a descriptive neighboring comparator chosen before inspecting its gap events; different layer roles may exist and it is not a matched causal control. 36→37,37→38,38→39 match conditional source-derived aux sampling locations but neither invocation nor tensor identity is established.',
 'window_sha256':sha(wp),'manifest_sha256':sha(mp),'kernel_sha256_by_rank':hashes,'samples':rows,'summary':summary,
 'limitations':['Times are relative to each rank-local gap; no cross-rank time alignment or task/request attribution.',
  'Overlapping stream row intervals do not prove work type, execution resource, dependency or source identity.',
  'Numeric and unknown stream may refer to duplicate/synthetic records; union avoids double-counting time but does not identify physical work.',
  'A gap not covered by exported rows is not necessarily idle; PMU, other processes, host work or events may be absent.',
  'Layer bounds are bracketing by source role invariants, not proof whole modules end at those exact points.']}
assert out['window_sha256']=='d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'
assert out['manifest_sha256']=='45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'
print(json.dumps(out,ensure_ascii=False,indent=2))
