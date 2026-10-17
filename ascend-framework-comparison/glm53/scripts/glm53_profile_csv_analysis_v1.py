"""Bounded stdlib analysis of native CSVs. Durations/intervals are profiler evidence, not baseline timing."""
import collections,csv,hashlib,json,math,re
from pathlib import Path

COMM=re.compile(r'hcom|hccl|all.?to.?all|all.?gather|all.?reduce|reduce.?scatter',re.I)
def number(value):
 try:result=float(str(value).replace(',','').strip())
 except (TypeError,ValueError):return None
 return result if math.isfinite(result) else None
def union(intervals):
 if not intervals:return 0.0
 ordered=sorted(intervals);start,end=ordered[0];total=0.0
 for left,right in ordered[1:]:
  if left>end:total+=end-start;start,end=left,right
  else:end=max(end,right)
 return total+end-start
def timing(rows,start_key,end_key,duration_key,name_key,type_key):
 all_intervals=[];comm=[];other=[];types=collections.Counter();names=collections.Counter();counts=collections.Counter();invalid=[];rows_seen=0
 for row in rows:
  rows_seen+=1;assert rows_seen<=2_000_000
  start=number(row[start_key]);duration=number(row[duration_key]);end=number(row[end_key]) if end_key else (start+duration if start is not None and duration is not None else None)
  if start is None or end is None or duration is None or end<start or duration<0:
   if len(invalid)<8:invalid.append({k:row.get(k) for k in [start_key,duration_key,name_key,type_key]})
   counts['invalid_rows']+=1;continue
  name=row[name_key];kind=row[type_key];types[kind]+=duration;names[name]+=duration;counts['valid_rows']+=1
  if abs(end-start-duration)>max(1.0,duration*.001):counts['duration_end_mismatch']+=1
  if end>start:
   pair=(start,end);all_intervals.append(pair);(comm if COMM.search(name+' '+kind) else other).append(pair)
 assert all_intervals,'No valid native task intervals'
 start=min(x[0] for x in all_intervals);end=max(x[1] for x in all_intervals);window=end-start;total=union(all_intervals);ucomm=union(comm);uother=union(other)
 return {'rows':rows_seen,'counts':dict(counts),'invalid_examples':invalid,'window_us':window,'recorded_interval_union_us':total,'communication_name_interval_union_us':ucomm,'other_task_interval_union_us':uother,'communication_other_overlap_us':max(0.0,ucomm+uother-total),'uncovered_within_recorded_window_us':max(0.0,window-total),'coverage_fraction':total/window if window else None,'native_type_duration_sums_us':dict(types),'top_names_duration_sums_us':[{'name':n,'duration_sum_us':t} for n,t in names.most_common(20)],'scope':'Union of recorded native intervals; communication classification is name/type based. Waiting tasks, overlapping streams and omitted task categories preclude equating these directly to physical NPU utilization or pure computation. Sums can overlap; no cross-rank duration addition.'}
def checked_files(root,expected):
 result={}
 for name,item in expected.items():
  path=root/name;assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve());stat=path.stat();assert stat.st_size==item['bytes']<=256*1024**2
  digest=hashlib.sha256()
  with path.open('rb') as file:
   for chunk in iter(lambda:file.read(1024**2),b''):digest.update(chunk)
  assert digest.hexdigest()==item['sha256'];assert path.stat().st_mtime_ns==stat.st_mtime_ns;result[name]=path
 return result
def csv_rows(path,columns):
 with path.open(newline='') as file:
  reader=csv.DictReader(file);assert set(columns)<=set(reader.fieldnames)
  yield from reader
def analyze_worker(root,settings):
 paths=checked_files(root,settings['files']);names={Path(p).name:p for p in paths};out={'namespace_pid':settings['namespace_pid'],'native_device_id':settings['device_id'],'TP_rank':settings['TP_rank'],'source_files':settings['files']}
 info=json.loads(paths[names[settings['profiler_info_name']]].read_text());assert info['rank_id']==settings['TP_rank'];start=info['start_info']['start_monotonic'];end=info['end_info']['MonotonicTimeEnd'];assert type(start) is int and type(end) is int and end>start;out['native_collection_duration_seconds']=(end-start)/1e9
 path=paths[names['task_time.csv']];out['all_native_tasks']=timing(csv_rows(path,['kernel_name','kernel_type','task_time(us)','task_start(us)','task_stop(us)']),'task_start(us)','task_stop(us)','task_time(us)','kernel_name','kernel_type')
 path=paths[names['kernel_details.csv']];out['native_kernel_details']=timing(csv_rows(path,['Name','Type','Start Time(us)','Duration(us)']),'Start Time(us)',None,'Duration(us)','Name','Type')
 stats=[]
 for row in csv_rows(paths[names['op_statistic.csv']],['OP Type','Core Type','Count','Total Time(us)']):
  duration=number(row['Total Time(us)']);count=number(row['Count'])
  if duration is not None and count is not None:stats.append({'op_type':row['OP Type'],'core_type':row['Core Type'],'count':count,'duration_sum_us':duration})
 out['top_native_operator_types']=sorted(stats,key=lambda x:x['duration_sum_us'],reverse=True)[:25]
 host=[]
 for row in csv_rows(paths[names['operator_details.csv']],['Name','Host Self Duration(us)','Host Total Duration(us)','Device Self Duration(us)']):
  value=number(row['Host Self Duration(us)'])
  if value is not None:host.append({'name':row['Name'],'host_self_us':value,'host_total_us':number(row['Host Total Duration(us)']),'device_self_us':number(row['Device Self Duration(us)'])})
 out['top_host_operator_self_time']=sorted(host,key=lambda x:x['host_self_us'],reverse=True)[:25]
 api=[]
 for row in csv_rows(paths[names['api_statistic.csv']],['Level','API Name','Time(us)','Count']):
  value=number(row['Time(us)'])
  if value is not None:api.append({'level':row['Level'],'name':row['API Name'],'time_sum_us':value,'count':number(row['Count'])})
 out['top_native_API_time']=sorted(api,key=lambda x:x['time_sum_us'],reverse=True)[:20]
 steps=list(csv_rows(paths[names['step_trace_time.csv']],['Step','Computing','Communication','Free','Stage']));assert len(steps)<=1024;out['native_step_statistics_rows']=steps;out['step_statistics_unit_scope']='Original native columns have no explicit time units; preserved verbatim, no guessed conversion.'
 return out
