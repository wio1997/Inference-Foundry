#!/usr/bin/env python3
"""Run98 low-overhead 8-rank aligned cycle stage spread audit."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[json.loads((root/f'evidence/20260924_loop036_metadata/run98/dag/rank{r}.json').read_text())['runtime_stage_ms'] for r in range(8)]
assert all(len(x)==300 for x in rows)

def pct(v,q):return sorted(v)[round((len(v)-1)*q)]
stages={}
for name in ('target','proposer','derived_target_metadata','acceptance','prepare_target'):
 per=[]
 for cycle in range(20,280):
  values=[rows[r][cycle][name] for r in range(8)]
  per.append({'cycle':cycle,'min_ms':min(values),'max_ms':max(values),'spread_ms':max(values)-min(values)})
 spreads=[x['spread_ms'] for x in per]
 stages[name]={'rank_cycle_spread_ms':{'median':statistics.median(spreads),'p90':pct(spreads,.9),'p99':pct(spreads,.99),'max':max(spreads)},'rank_cycle_median_of_max_ms':statistics.median(x['max_ms'] for x in per),'rank_cycle_median_of_min_ms':statistics.median(x['min_ms'] for x in per),'largest_spreads':sorted(per,key=lambda x:x['spread_ms'],reverse=True)[:5]}
result={'run':'run195','source':'Run98 low-overhead 300 cycle, 8-rank event DAG; steady ordinal cycles20-279','stages':stages,'interpretation':'Per-cycle rank duration spread is measurable without Run107 full profiler/synchronization. It is still not a causal removable latency: starts/ends are not cross-rank aligned, event duration can contain peer wait, and target graph required work is large.','limits':['Separate Run98 service, not same-state Run107.','Only duration spread, not absolute entry skew or root-cause decomposition.','No intervention or formal E2E.']}
out=root/'evidence/20260925_loop050_target_dependency/run195/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v['rank_cycle_spread_ms'] for k,v in stages.items()},indent=2))
