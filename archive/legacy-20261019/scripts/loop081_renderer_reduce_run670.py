"""Formal renderer-worker comparison, with trajectory kept separate."""
import hashlib,json,statistics
from pathlib import Path
root=Path('evidence/20260929_loop081_bound/run670/live');arms={}
for arm in ('workers1','workers4'):
 p=root/arm;rows=[]
 loaded=json.loads((p/'loaded_renderer.json').read_text());assert loaded['expected_workers']==int(arm[-1])
 for rep in range(1,4):
  b=json.loads((p/f'bench48_{rep}.json').read_text());s=b['summary'];assert s['n']==s['success']==48 and s['fail']==0
  assert all(x['output_tokens']==1024 and x['error'] is None for x in b['requests'])
  rr=[json.loads((p/'runtime'/f'rank0_cohort{c}.json').read_text()) for c in range(rep*4+1,rep*4+5)]
  cycles=sum(x['cycles'] for x in rr);runtime=sum(x['wall_seconds'] for x in rr)
  rows.append({'repeat':rep,'client_s':s['duration_s'],'output_tps':s['output_tps'],'cycles':cycles,'runtime_s':runtime,'runtime_ms_per_cycle':runtime/cycles*1000,'client_minus_runtime_s':s['duration_s']-runtime,'product_tokens_per_runtime_cycle':49152/cycles,'cohort_cycles':[x['cycles'] for x in rr]})
 arms[arm]={'rows':rows,'median_tps':statistics.median(x['output_tps'] for x in rows)}
cleanup=dict(x.split('=',1) for x in (root/'cleanup_status.txt').read_text().splitlines());assert set(cleanup.values())=={'0'}
for kind in ('source','scripts'):assert (root/f'{kind}_before.sha256').read_bytes()==(root/f'{kind}_after.sha256').read_bytes()
out={'arms':arms,'relative_median_tps_percent':100*(arms['workers4']['median_tps']/arms['workers1']['median_tps']-1),'cleanup':cleanup,'limits':'Separate launches and batching may change acceptance trajectory. Residual includes all non-fixed-serving Product work; useful tokens/cycle ratio includes prebulk outputs. No exact generated-text parity claim.'}
(root.parent/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
