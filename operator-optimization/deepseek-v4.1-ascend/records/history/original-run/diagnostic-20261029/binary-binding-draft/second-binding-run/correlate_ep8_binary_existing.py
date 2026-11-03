"""Read-only, bounded conditional C-recorder versus profiler API interval audit."""
from collections import Counter,defaultdict
from pathlib import Path
import json,hashlib
base=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029')
meta=base/'binding-diagnostic-selective2/runtime/metadata'
p=meta/'binding-trace.jsonl';st0=p.stat();raw=p.read_bytes();st1=p.stat();assert (st0.st_size,st0.st_mtime_ns)==(st1.st_size,st1.st_mtime_ns);rows=[json.loads(x) for x in raw.splitlines()]
side=meta/'events-746.jsonl';st0=side.stat();side_raw=side.read_bytes();st1=side.stat();assert (st0.st_size,st0.st_mtime_ns)==(st1.st_size,st1.st_mtime_ns);side_rows=[json.loads(x) for x in side_raw.splitlines()]
clock=[(r['time_ns']-r['perf_counter_ns']) for r in side_rows if isinstance(r.get('time_ns'),int) and isinstance(r.get('perf_counter_ns'),int)]
assert clock
# Profiler queue audit; timestamps are profiler epoch-labelled values, not C monotonic.
targets=[{'call_id':'746-20','device_task':47465,'api_start_ns':1793250235518395350,'api_end_ns':1793250235518408560},{'call_id':'746-65','device_task':50913,'api_start_ns':1793250238523157060,'api_end_ns':1793250238523170880}]
candidate_rows=[r for r in rows if r.get('pid')==746 and r.get('tid')==6857]
groups=defaultdict(list)
for r in candidate_rows:
 if r.get('event') in ('target_launch_begin','target_launch_end'):groups[r.get('call')].append(r)
pairs=[];bad=0
for call,events in groups.items():
 a=[x for x in events if x['event']=='target_launch_begin'];b=[x for x in events if x['event']=='target_launch_end']
 if len(a)!=1 or len(b)!=1 or a[0].get('func')!=b[0].get('func') or a[0].get('stream')!=b[0].get('stream') or b[0].get('rc')!=0 or a[0]['monotonic_ns']>b[0]['monotonic_ns']:
  bad+=1;continue
 pairs.append((a[0],b[0]))
# ±1ms is an explicitly unproved exploratory margin, NOT a hard bound.
lo=min(clock)-1_000_000;hi=max(clock)+1_000_000
for t in targets:
 found=[]
 for a,b in pairs:
  # There exists an offset under which the recorded C wrapper brackets the profiler API.
  possible_lo=max(lo,t['api_end_ns']-b['monotonic_ns']);possible_hi=min(hi,t['api_start_ns']-a['monotonic_ns'])
  if possible_lo<=possible_hi:
   found.append({'call':a['call'],'begin_ns':a['monotonic_ns'],'end_ns':b['monotonic_ns'],'offset_interval_ns':[possible_lo,possible_hi],'func':a.get('func'),'bin':a.get('bin'),'entry':a.get('entry'),'generation':a.get('generation'),'pre_call_sha256':a.get('pre_call_sha256'),'kind':a.get('kind'),'state_incomplete':a.get('state_incomplete'),'dropped_begin':a.get('dropped_since_last'),'dropped_end':b.get('dropped_since_last')})
 t['conditional_candidates']=found
result={'binding_sha256':hashlib.sha256(raw).hexdigest(),'binding_bytes':len(raw),'sidecar_sha256':hashlib.sha256(side_raw).hexdigest(),'sidecar_events':len(side_rows),'sidecar_offset_range_ns':[min(clock),max(clock)],'assumed_offset_range_ns':[lo,hi],'assumed_offset_margin_ns':1000000,'identity_assumption':'C pid 746/tid 6857 corresponds to profiler globalTid 3204045609673; not independently proved','clock_assumption':'same recorder monotonic clock and DB wall clock mapping remains inside chosen range; no hard guarantee','candidate_event_counts':dict(Counter(x.get('event') for x in candidate_rows)),'pid_tid_complete_pairs':len(pairs),'pid_tid_incomplete_groups':bad,'pid_tid_dropped_since_last_sum':sum(x.get('dropped_since_last',0) for x in candidate_rows),'pid_tid_stats_tail':[{k:r.get(k) for k in ('reason','state_incomplete','state_lock_drops','missing_bin','pending_drops','mutation_overlaps','launch_mutation_overlaps')} for r in candidate_rows if r.get('event')=='selective_stats'][-1:],'targets':targets,'strict_conclusion':'UNKNOWN: observed intervals and SHA are candidates, not binding; clock limits, losses, reuse and graph are not proven'}
print(json.dumps(result,indent=2))
