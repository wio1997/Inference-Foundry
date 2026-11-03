#!/usr/bin/env python3
"""Aggregate frozen model-49 main-stream task types on 48 fixed target spans; CPU only."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict,Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);args=ap.parse_args()
t=Path(args.task_root);trace=Path(args.trace_root)
inputs={
 'windows':('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'),
 'manifest':('evidence/TRACE-EXPORT-SHA256.json','45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'),
 'comparison':('agents/diagnosis/target-span-cost-comparison.json','e2c2ab268a3d8f09b83b721ce835244ea3ce8ae3ba0c59174bcd4052c836b840')}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,(path,digest) in inputs.items():
 if sha(t/path)!=digest:raise RuntimeError('frozen input sha mismatch: '+name)
read=lambda key:json.loads((t/inputs[key][0]).read_text())
windows=defaultdict(list)
for w in read('windows'):
 if w['run_in_model'] in (1,2,3):windows[(w['rank'],w['run_in_model'])].append(w)
assert len(windows)==48
prior={(x['rank'],x['run']):x for x in read('comparison')['rows']};assert set(windows)==set(prior)
spans=defaultdict(list)
for (rank,run),ws in windows.items():
 ws=sorted(ws,key=lambda w:w['layer']);assert [w['layer'] for w in ws]==list(range(40))
 stream=str(ws[0]['main_stream_id']);model=str(ws[0]['model_id']);assert model=='49'
 assert all(str(w['model_id'])==model and str(w['main_stream_id'])==stream for w in ws)
 a,b=ws[0]['window_start_ns'],ws[-1]['window_end_ns'];assert b-a==prior[(rank,run)]['span_ns']
 spans[rank].append((run,a,b,model,stream))
expected={x['rank']:x['kernel_csv_sha256'] for x in read('manifest')['exports']}
assert len(expected)==16 and set(spans)==set(expected)

def union(intervals):
 merged=[]
 for a,b in sorted(intervals):
  if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
  else:merged.append((a,b))
 return sum(b-a for a,b in merged)
rows=[];issues=[];exclude=Counter();unattributed=Counter();actual={}
for rank in sorted(expected):
 p=trace/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';actual[rank]=sha(p)
 if actual[rank]!=expected[rank]:raise RuntimeError('kernel CSV sha mismatch: '+rank)
 groups={run:defaultdict(list) for run,_,_,_,_ in spans[rank]};counts={run:Counter() for run,_,_,_,_ in spans[rank]}
 with p.open(newline='') as f:
  reader=csv.DictReader(f);required={'Type','Model ID','Stream ID','Start Time(us)','Duration(us)'}
  if not required.issubset(reader.fieldnames):raise RuntimeError('missing required columns '+rank)
  for z in reader:
   relevant=[x for x in spans[rank] if z['Model ID']==x[3] and z['Stream ID']==x[4]]
   if not relevant:continue
   typ=z['Type'].strip()
   if not typ:exclude['blank_type']+=1;continue
   if 'metadata' in typ.lower():exclude['metadata_type']+=1;continue
   try:
    a=Decimal(z['Start Time(us)'].strip())*1000
    if not a.is_finite():raise ValueError('nonfinite start')
   except (InvalidOperation,ValueError):
    # Without a start timestamp this row cannot be assigned in or outside a span.
    unattributed['invalid_start']+=1;continue
   try:
    d=Decimal(z['Duration(us)'].strip())*1000
    if not d.is_finite():raise ValueError('nonfinite duration')
   except (InvalidOperation,ValueError):
    matching=[run for run,start,end,_,_ in relevant if start<=a<end]
    if matching:
     for run in matching:issues.append({'rank':rank,'run':run,'type':typ,'reason':'invalid_duration_in_span'})
    elif any(a<end for _,_,end,_,_ in relevant):
     # A row starting before a span with unknown duration could intersect it.
     unattributed['invalid_duration_possible_overlap']+=1
    continue
   if d==0:
    if any(start<=a<end for _,start,end,_,_ in relevant):exclude['zero_duration_in_span']+=1
    continue
   if d<0:
    for run,start,end,_,_ in relevant:
     if start<=a<end:issues.append({'rank':rank,'run':run,'type':typ,'reason':'negative_duration_in_span'})
    continue
   b=a+d
   matching=[x for x in relevant if a<x[2] and b>x[1]]
   if not matching:continue
   if a!=a.to_integral_value() or d!=d.to_integral_value():
    for run,_,_,_,_ in matching:issues.append({'rank':rank,'run':run,'type':typ,'reason':'subnanosecond_interval_in_span'})
    continue
   a=int(a);b=int(b)
   for run,start,end,_,_ in matching:
    groups[run][typ].append((max(a,start),min(b,end)));counts[run][typ]+=1
 for run,start,end,_,_ in spans[rank]:
  d={typ:union(ints) for typ,ints in groups[run].items()};allints=[iv for ints in groups[run].values() for iv in ints]
  mc2=d.get('DispatchFFNCombine',0);p=prior[(rank,run)]['duration_ns']['mc2']
  if mc2!=p:issues.append({'rank':rank,'run':run,'reason':'mc2_union_vs_previous_mismatch','actual_ns':mc2,'previous_ns':p})
  rows.append({'rank':rank,'run':run,'span_ns':end-start,'type_union_ns':d,'type_event_count':dict(counts[run]),
    'all_types_union_ns':union(allints),'sum_type_unions_ns':sum(d.values()),'cross_type_overlap_excess_ns':sum(d.values())-union(allints)})
assert len(rows)==48
summ=[]
for run in (1,2,3):
 group=[r for r in rows if r['run']==run];types=sorted(set().union(*(r['type_union_ns'].keys() for r in group)))
 stats=[]
 for typ in types:
  values=[r['type_union_ns'].get(typ,0)/r['span_ns'] for r in group]
  stats.append({'type':typ,'rank_count_observed':sum(typ in r['type_union_ns'] for r in group),
   'fraction_min':min(values),'fraction_median':statistics.median(values),'fraction_max':max(values)})
 stats.sort(key=lambda x:x['fraction_median'],reverse=True)
 summ.append({'run':run,'types':stats,'all_types_coverage_median':statistics.median(r['all_types_union_ns']/r['span_ns'] for r in group),
 'cross_type_overlap_excess_median':statistics.median(r['cross_type_overlap_excess_ns']/r['span_ns'] for r in group)})
print(json.dumps({'status':'COMPLETE' if not issues and not unattributed else 'INCOMPLETE','scope':'16 ranks x 3 frozen target layer0-to-layer39 spans, model 49, exact recorded main stream; task Type labels only',
 'input_sha256':{k:v[1] for k,v in inputs.items()},'kernel_sha256_by_rank':actual,'issues':issues,'unattributed_invalid_time_counts':dict(unattributed),'excluded_matching_stream_rows':dict(exclude),'per_run':summ,'rows':rows,
 'limits':['Type is an exporter label, not guaranteed operator/module or a unique kernel instance. Only positive-duration records of selected model and stream are included; names containing metadata and blank types are excluded and counted.',
 'Time intervals are clipped at span boundaries and unioned within each label; different labels can overlap or nest; type fractions cannot be summed as exclusive costs.',
 'All-types union is selected exported records coverage, not device utilization or device idle; other streams, models, uncaptured tasks or host work are excluded.',
 'Rank clocks stay separate; per-run distributions summarize 16 rank-local fractions, and runs are not proven matching requests.',
 'Model 49 main stream target-layer span is not end-to-end service time, an exhaustive framework ranking, a causal bottleneck ranking or achievable speedup.',
 'Bad start or bad duration that could overlap a span is counted as unassigned and prevents a COMPLETE status; zero-duration rows whose start is in a span are counted and omitted.',
 'Zero means label absent in selected exported rows for that rank/run, not proof of absence of device work.'], 'no_device_work':True},ensure_ascii=False,indent=2))
