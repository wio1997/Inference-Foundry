#!/usr/bin/env python3
"""Rank-local exported events in gaps between 40 captured target layer windows."""
import argparse,bisect,csv,hashlib,json,statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);wp=T/'captured-layer-attribution/module-call-metrics.json';mp=T/'evidence/TRACE-EXPORT-SHA256.json'
expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']}
by=defaultdict(list)
for w in json.loads(wp.read_text()):
 if w['run_in_model'] in (1,2,3):by[(w['rank'],w['run_in_model'])].append(w)
assert len(by)==48 and len({k[0] for k in by})==16 and {k[0] for k in by}==set(expected)
gaps=defaultdict(list);span_ns={};layer_ns={}
for (rank,run),ws in by.items():
 ws.sort(key=lambda w:w['window_start_ns']);assert len(ws)==40 and [w['layer'] for w in ws]==list(range(40))
 assert len({str(w['model_id']) for w in ws})==1 and len({str(w['main_stream_id']) for w in ws})==1
 assert all(x['window_end_ns']<=y['window_start_ns'] for x,y in zip(ws,ws[1:]))
 span_ns[(rank,run)]=ws[-1]['window_end_ns']-ws[0]['window_start_ns'];layer_ns[(rank,run)]=sum(w['window_end_ns']-w['window_start_ns'] for w in ws)
 for i in range(39):
  s,e=ws[i]['window_end_ns'],ws[i+1]['window_start_ns']
  if s<e:gaps[rank].append({'start':s,'end':e,'run':run,'after_layer':i})
for rank in gaps:
 gaps[rank].sort(key=lambda x:x['start']);assert all(x['end']<=y['start'] for x,y in zip(gaps[rank],gaps[rank][1:]))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def union(v):
 merged=[]
 for a,b in sorted(v):
  if not merged or a>merged[-1][1]:merged.append([a,b])
  else:merged[-1][1]=max(merged[-1][1],b)
 return sum(b-a for a,b in merged)
events=defaultdict(list);hashes={}
for rank,gs in sorted(gaps.items()):
 p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';hashes[rank]=sha(p);assert hashes[rank]==expected[rank]
 starts=[x['start'] for x in gs];ends=[x['end'] for x in gs]
 with p.open() as f:
  for z in csv.DictReader(f):
   start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
   if end<=start:continue
   for i in range(bisect.bisect_right(ends,start),bisect.bisect_left(starts,end)):
    g=gs[i];lo=max(start,g['start']);hi=min(end,g['end'])
    if lo<hi:events[(rank,g['run'])].append({'start_ns':lo,'end_ns':hi,'type':z['Type'],'stream':z['Stream ID'].strip(),'model':z['Model ID'].strip(),'after_layer':g['after_layer']})
assert len(hashes)==16
rows=[]
for (rank,run),ws in sorted(by.items()):
 gs=[g for g in gaps[rank] if g['run']==run];gap_ns=sum(g['end']-g['start'] for g in gs)
 assert span_ns[(rank,run)]==layer_ns[(rank,run)]+gap_ns
 ev=events[(rank,run)];assert len(ev)<=5000,'unexpectedly large gap event set'
 numeric=[x for x in ev if x['stream'].isdigit()];unknown=[x for x in ev if not x['stream'].isdigit()]
 spans=lambda e:[(x['start_ns'],x['end_ns']) for x in e]
 bytype=defaultdict(list)
 for e in numeric:bytype[e['type']].append((e['start_ns'],e['end_ns']))
 tasktypes=sorted([{'type':k,'clipped_rows':len(v),'union_ns':union(v)} for k,v in bytype.items()],key=lambda x:-x['union_ns'])
 numeric_ns=union(spans(numeric));unknown_ns=union(spans(unknown));assert numeric_ns<=gap_ns and unknown_ns<=gap_ns
 rows.append({'rank':rank,'run':run,'forward_bracket_ns':span_ns[(rank,run)],'captured_layer_windows_ns':layer_ns[(rank,run)],
   'inter_layer_gap_ns':gap_ns,'numeric_stream_union_ns':numeric_ns,'unknown_stream_union_ns':unknown_ns,
   'gap_no_numeric_stream_row_ns':gap_ns-numeric_ns,'numeric_clipped_rows':len(numeric),'unknown_clipped_rows':len(unknown),
   'numeric_type_unions':tasktypes,'largest_gaps':[{'after_layer':g['after_layer'],'duration_ns':g['end']-g['start']} for g in sorted(gs,key=lambda x:-(x['end']-x['start']))[:5]]})
summ=[]
for run in (1,2,3):
 rs=[r for r in rows if r['run']==run];assert len(rs)==16
 names={x['type'] for r in rs for x in r['numeric_type_unions']}
 type_summary=[]
 for name in names:
  vals=[next((e['union_ns'] for e in r['numeric_type_unions'] if e['type']==name),0) for r in rs]
  type_summary.append({'type':name,'nonzero_ranks':sum(v>0 for v in vals),'median_union_us':statistics.median(vals)/1000,
    'max_union_us':max(vals)/1000})
 summ.append({'run':run,'rank_count':16,'forward_bracket_us_median':statistics.median(r['forward_bracket_ns'] for r in rs)/1000,
    'inter_layer_gap_us_median':statistics.median(r['inter_layer_gap_ns'] for r in rs)/1000,
    'numeric_stream_union_us_median':statistics.median(r['numeric_stream_union_ns'] for r in rs)/1000,
    'gap_no_numeric_row_us_median':statistics.median(r['gap_no_numeric_stream_row_ns'] for r in rs)/1000,
    'top_numeric_type_unions_by_median':sorted(type_summary,key=lambda x:(-x['median_union_us'],x['type']))[:10],
    'top_numeric_type_unions_by_max':sorted(type_summary,key=lambda x:(-x['max_union_us'],x['type']))[:10]})
out={'scope':'Each of 16 ranks and 3 captured target runs; gap is only between 40 source-attributed layer windows, not end-to-end or module ownership.',
 'window_sha256':sha(wp),'manifest_sha256':sha(mp),'kernel_sha256_by_rank':hashes,'samples':rows,'run_summary':summ,
 'limitations':['Intervals may contain unrelated or auxiliary work; no ownership, request identity or causality is established by overlap.',
  'Numeric stream and unknown-stream rows are separated; HCOM names may still represent nested/summarizing rows.',
  'Per-type unions can overlap each other and cannot be added or called exclusive time.',
  'Gaps not covered by exported numeric stream rows are not proven device idle; no cross-rank absolute-clock comparison.',
  'Layer brackets omit work before layer0 and after layer39 and are not entire request or forward time.']}
print(json.dumps(out,ensure_ascii=False,indent=2))
