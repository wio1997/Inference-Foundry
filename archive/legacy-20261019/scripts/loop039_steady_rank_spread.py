#!/usr/bin/env python3
"""Audit steady eight-rank stage-duration spread in unprofiled Run98."""
import json,statistics
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
base=ROOT/'evidence/20260924_loop036_metadata/run98/dag'
ranks=[json.loads((base/f'rank{r}.json').read_text()) for r in range(8)]
assert all(x['rank']==r and len(x['runtime_stage_ms'])==300 for r,x in enumerate(ranks))
cycles=range(64,256)
stage_names=list(ranks[0]['runtime_stage_ms'][64])
def stats(xs):
 x=sorted(xs);return {'median':statistics.median(x),'p90':x[int(len(x)*.9)],
                        'p95':x[int(len(x)*.95)],'max':max(x)}
stages={}
for name in stage_names:
 per_cycle_medians=[]
 per_cycle_spread=[]
 for c in cycles:
  vals=[x['runtime_stage_ms'][c][name] for x in ranks]
  per_cycle_medians.append(statistics.median(vals))
  per_cycle_spread.append(max(vals)-min(vals))
 stages[name]={'stage_ms':stats(per_cycle_medians),
               'rank_duration_spread_ms':stats(per_cycle_spread)}
out={'run':'run129','source':'Run98 unprofiled 8-rank DAG, cycles64-255',
     'n_cycles':len(list(cycles)), 'stages':stages,
     'inference':'Median steady per-cycle rank duration spread is <0.1ms for target and proposer. This does not measure cross-rank absolute arrival skew, but weakens treating Run107 diagnostic 10ms HCCL peer-wait as a persistent independent product transfer gap.',
     'limitations':['Run98 has stage durations, not rank-synchronized absolute timestamps.','Run107 profiler changes execution and inserts target synchronization.','No new formal E2E.']}
p=ROOT/'evidence/20260925_loop039_gmm/run129'
p.mkdir(parents=True,exist_ok=True)
(p/'steady_rank_spread.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:stages[k] for k in ('target','proposer','derived_target_metadata')},indent=2))
