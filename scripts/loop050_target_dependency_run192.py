#!/usr/bin/env python3
"""Read-only Run107 first collective and remainder dependency attribution."""
import glob,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prior=json.loads((root/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in prior['excluded_windows']}
windows=[]
for rank in range(8):
 p=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
 assert len(p)==1
 events=json.loads(Path(p[0]).read_text())
 scopes=sorted((e for e in events if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
 assert len(scopes)==2
 for cycle,scope in zip((64,65),scopes):
  if (rank,cycle) in excluded:continue
  start=float(scope['ts']);end=start+float(scope['dur'])
  kernels=sorted(({'name':e['name'],'start':float(e['ts']),'end':float(e['ts'])+float(e['dur']),'dur':float(e['dur'])} for e in events if start<=float(e.get('ts',-1))<end and e.get('args',{}).get('Task Type','').startswith('KERNEL_')),key=lambda x:x['start'])
  comm=[e for e in kernels if e['name'].startswith('hcom_')]
  assert len(comm)==260
  first=comm[0]
  preceding=[e for e in kernels if e['start']<first['start'] and not e['name'].startswith('hcom_')]
  intersect=[e for e in preceding if e['end']>first['start']]
  windows.append({'rank':rank,'cycle':cycle,'first_rs_ms':first['dur']/1000,'first_rs_start_us':first['start'],'first_rs_end_us':first['end'],'first_rs_offset_from_scope_ms':(first['start']-start)/1000,'pre_first_noncomm_count':len(preceding),'pre_first_latest_kernel':max(preceding,key=lambda e:e['end'])['name'] if preceding else None,'pre_first_overlap_count':len(intersect),'pre_first_overlap_latest_end_ms':max((e['end']-first['start'])/1000 for e in intersect) if intersect else 0,'remaining_comm_sum_ms':sum(e['dur'] for e in comm[1:])/1000,'first_rs_duration_fraction_of_all_comm':first['dur']/sum(e['dur'] for e in comm)})
cycles={}
for cycle in (64,65):
 w=[x for x in windows if x['cycle']==cycle]
 cycles[str(cycle)]={'ranks':len(w),'first_rs_start_skew_ms':(max(x['first_rs_start_us'] for x in w)-min(x['first_rs_start_us'] for x in w))/1000,'first_rs_end_skew_ms':(max(x['first_rs_end_us'] for x in w)-min(x['first_rs_end_us'] for x in w))/1000,'first_rs_duration_ms_by_rank':{str(x['rank']):x['first_rs_ms'] for x in w},'first_rs_start_offset_ms_by_rank':{str(x['rank']):x['first_rs_offset_from_scope_ms'] for x in w}}
result={'run':'run192','source':'Run107 15 valid profiled/synchronized target windows','cycles':cycles,'windows':windows,'summary':{'first_rs_ms_median':statistics.median(x['first_rs_ms'] for x in windows),'remaining_comm_ms_median':statistics.median(x['remaining_comm_sum_ms'] for x in windows),'first_rs_overlapped_by_preceding_noncomm_windows':sum(bool(x['pre_first_overlap_count']) for x in windows)},'interpretation':'First HCCL reduce-scatter exhibits rank-arrival skew but nearly aligned completion (Run145); duration cannot be equated with fabric transfer. Remaining 259 HCCL tasks are a smaller ~5.2ms kernel sum. Within-rank preceding task audit screens whether first RS is delayed by compute, but cross-rank clock alignment and profiler/sync effects preclude a removable-time claim.','limits':['Profiler/sync perturbs the target stage; no formal E2E comparison.','Trace timestamps across ranks are used only diagnostically; no calibrated clock proof.','Kernel sum and overlap are not a causal critical-path saving.']}
out=root/'evidence/20260925_loop050_target_dependency/run192/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cycles':cycles,'summary':result['summary'],'pre_first_latest_kernel_names':sorted(set(x['pre_first_latest_kernel'] for x in windows))},indent=2))
