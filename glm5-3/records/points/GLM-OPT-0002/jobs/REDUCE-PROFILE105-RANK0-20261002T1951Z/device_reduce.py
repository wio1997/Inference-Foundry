from pathlib import Path
import json,csv,collections,sys
root=Path('/data/tiankuan/wio/glm52-pd/deploy/private/profile_run105_export_rank0');d=next(root.glob('PROF_*'));raw=Path('/data/tiankuan/wio/glm52-pd/deploy/private/profile_run105')
windows=[]
for prof in sorted(raw.glob('PROF_*')):
 dev=next(prof.glob('device_*'));a=json.loads(next(dev.glob('start_info.*')).read_text());b=json.loads(next(f for f in dev.glob('end_info.*')if not f.name.endswith('.done')).read_text());windows.append(dict(profile=prof.name,device=dev.name,start_us=int(a['collectionTimeBegin']),end_us=int(b['collectionTimeEnd']),raw_clock_start_ns=int(a['clockMonotonicRaw']),raw_clock_end_ns=int(b['clockMonotonicRaw'])))
lo=max(x['start_us']for x in windows);hi=min(x['end_us']for x in windows);assert hi>lo
def union(iv):
 v=sorted(iv);out=[]
 for a,b in v:
  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
  else:out.append([a,b])
 gaps=[];prev=lo
 for a,b in out:
  if a>prev:gaps.append([prev,a])
  prev=max(prev,b)
 if prev<hi:gaps.append([prev,hi])
 return dict(intervals=len(out),union_us=sum(b-a for a,b in out),fraction=sum(b-a for a,b in out)/(hi-lo),largest_gaps_us=sorted([b-a for a,b in gaps],reverse=True)[:20])
types=collections.defaultdict(lambda:dict(count=0,sum_us=0.,intervals=[]));streams=collections.defaultdict(list);seen=set();duplicates=0;total=0;first=None;last=None
f=next((d/'mindstudio_profiler_output').glob('task_time*.csv'))
with f.open()as z:
 for row in csv.DictReader(z):
  total+=1;a=float(row['task_start(us)']);b=float(row['task_stop(us)']);assert b>=a
  first=a if first is None else min(first,a);last=b if last is None else max(last,b)
  key=(row['stream_id'],row['task_id'],a,b,row['kernel_type'],row['kernel_name'])
  if key in seen:duplicates+=1;continue
  seen.add(key)
  a=max(a,lo);b=min(b,hi)
  if b<=a:continue
  t=types[row['kernel_type']];t['count']+=1;t['sum_us']+=b-a;t['intervals'].append((a,b));streams[row['stream_id']].append((a,b))
out=dict(common_device_window_us=[lo,hi],common_device_window_s=(hi-lo)/1e6,windows=windows,task_time_rows=total,exact_duplicate_rows=duplicates,task_trace_span_s=(last-first)/1e6,kernel_types={k:dict(count=v['count'],sum_us=v['sum_us'],**union(v['intervals']))for k,v in types.items()},all_task_union=union([x for v in types.values()for x in v['intervals']]),streams={k:union(v)for k,v in streams.items()})
ops=collections.defaultdict(lambda:dict(count=0,sum_us=0.,shapes=collections.Counter(),intervals=[]));excluded=0;seen=set()
f=next((d/'mindstudio_profiler_output').glob('op_summary*.csv'))
with f.open()as z:
 for row in csv.DictReader(z):
  if row['Stream ID']=='N/A'or row['Task ID']=='N/A':excluded+=1;continue
  a=float(row['Task Start Time(us)']);b=a+float(row['Task Duration(us)']);key=(row['Stream ID'],row['Task ID'],a,b)
  if key in seen:continue
  seen.add(key);a=max(a,lo);b=min(b,hi)
  if b<=a:continue
  v=ops[row['OP Type']];v['count']+=1;v['sum_us']+=b-a;v['shapes'][row['Input Shapes']]+=1;v['intervals'].append((a,b))
out.update(excluded_semantic_rows_without_task_identity=excluded,operators={k:dict(count=v['count'],sum_us=v['sum_us'],shapes=v['shapes'].most_common(4),**union(v['intervals']))for k,v in sorted(ops.items(),key=lambda x:-x[1]['sum_us'])},limits=['Rank0 only; common16device window derived from native epoch collectionTime values, raw clock/time mapping provenance recorded. No crosshost166/client-clock alignment assumed.','Union of reported device tasks is not hardware utilization or critical-path lowerbound; overlapping streams, waits and nested task records cannot be added as serialcost.','NoCurrent/KEEP/stablecapacity/ordinaryTPS; profiler overhead and rankcriticalchain attribution unresolved.'])
print(json.dumps(out))
