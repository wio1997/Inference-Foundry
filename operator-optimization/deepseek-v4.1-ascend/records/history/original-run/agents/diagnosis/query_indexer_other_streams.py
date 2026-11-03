#!/usr/bin/env python3
"""Bounded rank-local interval overlap from frozen kernel_details.csv, read only."""
import argparse,bisect,csv,hashlib,json,statistics
from decimal import Decimal
from pathlib import Path
from collections import defaultdict
ap=argparse.ArgumentParser();ap.add_argument('--trace-root',required=True);a=ap.parse_args()
frozen=json.loads(FROZEN_INPUT);trace=Path(a.trace_root)
by=defaultdict(list)
for x in frozen['targets']:by[x['rank']].append(x)
if len(by)!=16 or len(frozen['targets'])!=384:raise RuntimeError('target count')
for rank,ws in by.items():
 ws.sort(key=lambda w:w['start']);
 if len(ws)!=24 or any(a['end']>b['start'] for a,b in zip(ws,ws[1:])):raise RuntimeError('overlapping targets')
 if len({(w['layer'],w['run']) for w in ws})!=24:raise RuntimeError('duplicate targets')
 if any(not str(w['stream']).isdigit() for w in ws):raise RuntimeError('unknown target stream')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def union(spans):
 out=[]
 for a,b in sorted(spans):
  if a>=b:continue
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(out[-1][1],b)
 return out
def length(spans):return sum(b-a for a,b in union(spans))
def summarize(rows):
 groups=defaultdict(list)
 for z in rows:groups[(z['type'],z['core'])].append((z['start'],z['end']))
 return {'rows':len(rows),'union_ns':length([(z['start'],z['end']) for z in rows]),
  'categories':[{'type':k[0],'core':k[1],'row_count':len(v),'union_ns':length(v)} for k,v in sorted(groups.items())]}
known={'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'}
results=[];actual={}
for rank,ws in sorted(by.items()):
 p=trace/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';actual[rank]=sha(p)
 if actual[rank]!=frozen['csv_sha256_by_rank'][rank]:raise RuntimeError('CSV hash mismatch '+rank)
 ends=[w['end'] for w in ws];events=[[] for w in ws];own=[0]*len(ws)
 with p.open(newline='') as f:
  reader=csv.DictReader(f)
  need={'Type','Name','Model ID','Stream ID','Task ID','Accelerator Core','Start Time(us)','Duration(us)'}
  if not need.issubset(reader.fieldnames or ()):raise RuntimeError('missing fields')
  for z in reader:
   start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
   if end<=start:continue
   i=bisect.bisect_right(ends,start)
   while i<len(ws) and ws[i]['start']<end:
    w=ws[i];stream=z['Stream ID'].strip();typ=z['Type'].strip();name=z['Name'].strip();core=z['Accelerator Core'].strip()
    if start==w['start'] and end==w['end'] and typ=='VllmAscendQuantLightningIndexerV2' and stream==str(w['stream']) and z['Model ID'].strip()==str(w['model']):own[i]+=1
    if stream!=str(w['stream']):
     if len(typ)>128 or len(core)>64 or len(name)>256:raise RuntimeError('unexpected field length')
     category='unknown_stream' if not stream.isdigit() else ('communication_named' if typ.lower().startswith('hcom_') or name.lower().startswith('hcom_') else ('known_core_numeric' if core in known else 'unclassified_numeric'))
     events[i].append({'start':max(start,w['start']),'end':min(end,w['end']),'type':typ,'core':core,'category':category,'stream':stream,'model':z['Model ID'].strip(),'task_id':z['Task ID'].strip()})
    i+=1
 for i,w in enumerate(ws):
  if own[i]!=1:raise RuntimeError('target not unique '+rank)
  if len(events[i])>500:raise RuntimeError('bounded sample exceeded')
  # Intervals are clipped to target and unioned; exact-row dedup is not used as a time summation.
  cats={k:summarize([z for z in events[i] if z['category']==k]) for k in ('known_core_numeric','unclassified_numeric','communication_named','unknown_stream')}
  categories=[z['category'] for z in events[i]];dupes=len(categories)-len(set((z['start'],z['end'],z['stream'],z['model'],z['task_id'],z['type'],z['core']) for z in events[i]))
  numeric=length([(z['start'],z['end']) for z in events[i] if z['category'] in ('known_core_numeric','unclassified_numeric')]); duration=w['end']-w['start']
  if not (0<=numeric<=duration and all(0<=v['union_ns']<=duration for v in cats.values())):raise RuntimeError('bad union')
  results.append({'rank':rank,'layer':w['layer'],'run':w['run'],'target_ns':[w['start'],w['end']],'duration_ns':duration,
    'categories':cats,'numeric_excluding_hcom_union_ns':numeric,'exact_duplicate_rows':dupes,'clipped_rows':events[i]})
if len(results)!=384:raise RuntimeError('result count')
summary=[]
for layer in (2,8,14,20,24,28,32,36):
 for run in (1,2,3):
  xs=[v for v in results if (v['layer'],v['run'])==(layer,run)]
  if len(xs)!=16:raise RuntimeError('group count')
  s={'layer':layer,'run':run,'n':len(xs)}
  for cat in ('known_core_numeric','unclassified_numeric','communication_named','unknown_stream'):
   rr=[z['categories'][cat]['union_ns']/z['duration_ns'] for z in xs]
   s[cat]={'nonzero':sum(v>0 for v in rr),'fraction_min':min(rr),'fraction_median':statistics.median(rr),'fraction_max':max(rr)}
  summary.append(s)
print(json.dumps({'source':{'frozen_windows_sha256':frozen['window_result_sha256'],'paired_sha256':frozen['paired_sha256'],'kernel_csv_sha256_by_rank':actual},
 'scope':'16 ranks x 8 layers x 3 Indexer windows; all positively timed kernel CSV records intersected and clipped per rank',
 'core_allowlist':sorted(known),'samples':results,'summary':summary,
 'limits':['known-core is CSV label classification, not hardware counter or proof of execution/throughput; unclassified is retained separately',
 'hcom-prefixed numeric rows and nonnumeric streams are separate and excluded from numeric-other union as possible summaries/unknown identities',
 'clipped intervals are unioned, avoiding duplicate/overlap summation; category unions may overlap and must not be summed',
 'No row proves same request, dependency, network transfer, resource contention or actual latency hidden',
 'No matching exported rows do not prove idle device; capture coverage is not established; no cross-rank absolute-time comparison',
 'Unnamed non-hcom aggregates could remain among numeric labels; row identity/exporter contracts not established'], 'no_device_work':True},ensure_ascii=False,indent=2))
