#!/usr/bin/env python3
"""Read-only per-rank target entry and prior-stage timing in Run107."""
import json,glob,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[]
for rank in range(8):
 p=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
 assert len(p)==1
 e=json.loads(Path(p[0]).read_text())
 def scopes(name):return sorted([x for x in e if x.get('cat')=='cpu_op' and x.get('name')==name],key=lambda x:float(x['ts']))
 stage={name:scopes('extreme::'+name) for name in ('target','proposer','prepare_target','derived_target_metadata','acceptance')}
 assert all(len(v)==2 for v in stage.values())
 for idx,cycle in enumerate((64,65)):
  target=stage['target'][idx]
  start=float(target['ts']);end=start+float(target['dur'])
  comm=sorted([x for x in e if start<=float(x.get('ts',-1))<end and x.get('name')=='hcom_reduceScatter_' and x.get('args',{}).get('Task Type','').startswith('KERNEL_')],key=lambda x:float(x['ts']))
  assert len(comm)==87
  first=comm[0];first_end=float(first['ts'])+float(first['dur'])
  prior=stage['proposer'][idx-1] if idx else None
  def duration(x):return float(x['dur'])/1000
  rows.append({'rank':rank,'cycle':cycle,'target_start_us':start,'target_scope_ms':duration(target),'first_rs_start_us':float(first['ts']),'first_rs_end_us':first_end,'target_start_before_first_rs_end_ms':(first_end-start)/1000,'prepare_target_ms':duration(stage['prepare_target'][idx]),'target_metadata_ms':duration(stage['derived_target_metadata'][idx]),'previous_proposer_ms':duration(prior) if prior else None,'previous_proposer_end_to_target_start_ms':(start-(float(prior['ts'])+float(prior['dur'])))/1000 if prior else None})
cycles={}
for cycle in (64,65):
 w=[x for x in rows if x['cycle']==cycle and (x['rank'],cycle)!=(1,65)]
 cycles[str(cycle)]={'ranks':len(w),'target_entry_raw_skew_ms':(max(x['target_start_us'] for x in w)-min(x['target_start_us'] for x in w))/1000,'first_rs_end_raw_skew_ms':(max(x['first_rs_end_us'] for x in w)-min(x['first_rs_end_us'] for x in w))/1000,'target_entry_relative_to_first_rs_end_ms_by_rank':{str(x['rank']):round(x['target_start_before_first_rs_end_ms'],3) for x in w},'previous_proposer_ms_by_rank':{str(x['rank']):round(x['previous_proposer_ms'],3) for x in w} if cycle==65 else None,'previous_proposer_end_to_target_start_ms_by_rank':{str(x['rank']):round(x['previous_proposer_end_to_target_start_ms'],3) for x in w} if cycle==65 else None,'metadata_ms_by_rank':{str(x['rank']):round(x['target_metadata_ms'],3) for x in w}}
result={'run':'run193','source':'Run107 synchronized/profiler two target cycles per rank','cycles':cycles,'rows':rows,'interpretation':'Target entry skew closely matches the first reduce-scatter start skew because first collective starts at nearly constant offset after target entry. This locates peer wait upstream of the target graph first collective; the trace alone cannot prove whether previous proposer, host scheduling, graph replay, or synchronization controls the late rank.','limits':['Cross-rank timestamps are diagnostic, though aligned collective end supports approximate common clock.','Two profiled cycles, one excluded anomalous rank1 cycle65.','CPU stage durations are inclusive and profiler perturbed.']}
out=root/'evidence/20260925_loop050_target_dependency/run193/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(cycles,indent=2))
