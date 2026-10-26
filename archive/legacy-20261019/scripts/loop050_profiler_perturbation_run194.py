#!/usr/bin/env python3
"""Compare Run107 profiled proposer with Run98 low-overhead same frozen shape."""
import json,glob,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
profile=[];low=[]
for rank in range(8):
 p=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))[0]
 e=json.loads(Path(p).read_text())
 scopes=sorted((x for x in e if x.get('cat')=='cpu_op' and x.get('name')=='extreme::proposer'),key=lambda x:float(x['ts']))
 assert len(scopes)==2
 d=json.loads((root/f'evidence/20260924_loop036_metadata/run98/dag/rank{rank}.json').read_text())
 for idx,cycle in enumerate((64,65)):
  profile.append({'rank':rank,'cycle':cycle,'proposer_cpu_scope_ms':float(scopes[idx]['dur'])/1000})
  low.append({'rank':rank,'cycle':cycle,'proposer_event_ms':d['runtime_stage_ms'][cycle]['proposer'],'target_event_ms':d['runtime_stage_ms'][cycle]['target']})
P=[x['proposer_cpu_scope_ms'] for x in profile];L=[x['proposer_event_ms'] for x in low]
result={'run':'run194','source_profile':'Run107 two synchronized/profiler cycles','source_low_overhead':'Run98 300-cycle event DAG, same frozen Extreme shape but separate service','profile_proposer_cpu_scope_ms':{'median':statistics.median(P),'min':min(P),'max':max(P)},'low_overhead_proposer_event_ms_cycles64_65':{'median':statistics.median(L),'min':min(L),'max':max(L)},'profile_to_low_median_ratio':statistics.median(P)/statistics.median(L),'low_overhead_proposer_event_range_by_cycle':{str(c):{'min':min(x['proposer_event_ms'] for x in low if x['cycle']==c),'max':max(x['proposer_event_ms'] for x in low if x['cycle']==c)} for c in (64,65)},'profile':profile,'low_overhead':low,'decision':'Run107 proposer CPU scope is severely perturbed relative to Run98 event stages; its apparent 8-11ms rank imbalance and propagated first-collective wait cannot be promoted as a product gap. Run107 still provides kernel family order/sums but not an unprofiled cross-rank scheduling saving.','limits':['Different services and different timing primitives; ratio is perturbation warning, not precise profiler overhead.','Run98 event intervals are not same-state with Run107 trace.','No intervention or E2E gain measured.']}
out=root/'evidence/20260925_loop050_target_dependency/run194/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('profile_proposer_cpu_scope_ms','low_overhead_proposer_event_ms_cycles64_65','profile_to_low_median_ratio','low_overhead_proposer_event_range_by_cycle')},indent=2))
