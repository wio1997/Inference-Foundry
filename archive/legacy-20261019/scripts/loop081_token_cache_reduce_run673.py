"""Formal renderer-worker comparison, with trajectory kept separate."""
import hashlib,json,statistics
from pathlib import Path
root=Path('evidence/20260929_loop081_bound/run673/live');arms={}
for arm in ('cacheoff','cacheon'):
 p=root/arm;rows=[]
 loaded=json.loads((p/'loaded_renderer.json').read_text());assert loaded['expected_workers']==4
 for rep in range(1,4):
  b=json.loads((p/f'bench48_{rep}.json').read_text());s=b['summary'];assert s['n']==s['success']==48 and s['fail']==0
  assert all(x['output_tokens']==1024 and x['error'] is None for x in b['requests'])
  rr=[json.loads((p/'runtime'/f'rank0_cohort{c}.json').read_text()) for c in range(rep*4+1,rep*4+5)]
  cycles=sum(x['cycles'] for x in rr);runtime=sum(x['wall_seconds'] for x in rr)
  rows.append({'staged_accepted_tokens_per_cycle':sum(sum(x['staged_output_counts']) for x in rr)/cycles,'overshoot_tokens':sum(x['overshoot_tokens'] for x in rr),'acceptance_window_means':[x['acceptance_window_means'] for x in rr],'repeat':rep,'client_s':s['duration_s'],'output_tps':s['output_tps'],'cycles':cycles,'runtime_s':runtime,'runtime_ms_per_cycle':runtime/cycles*1000,'client_minus_runtime_s':s['duration_s']-runtime,'product_tokens_per_runtime_cycle':49152/cycles,'cohort_cycles':[x['cycles'] for x in rr]})
 arms[arm]={'rows':rows,'median_tps':statistics.median(x['output_tps'] for x in rows)}
for repeat in ('warmup','bench48_1','bench48_2','bench48_3'):
 a=json.loads((root/'cacheoff'/f'{repeat}.json').read_text())['requests']
 b=json.loads((root/'cacheon'/f'{repeat}.json').read_text())['requests']
 assert {x['i']:x['input_tokens'] for x in a}=={x['i']:x['input_tokens'] for x in b}
for arm in ('cacheoff','cacheon'):
 files=list((root/arm/'runtime').glob('rank*_cohort*.json'));assert len(files)==128
 seen={}
 for f in files:
  r=json.loads(f.read_text());assert r['pass'] and r['host_mirror_exact'] and 'FULL' in r['target_graph_mode']
  assert r['generated_output_counts']==[1024]*12
  key=(r['rank'],r['cohort']);assert key not in seen;seen[key]=r['cycles']
 assert set(seen)=={(r,c) for r in range(8) for c in range(1,17)}
 for c in range(1,17):assert len({seen[(r,c)] for r in range(8)})==1
 for c in range(1,17):
  rankrows=[json.loads((root/arm/'runtime'/f'rank{r}_cohort{c}.json').read_text()) for r in range(8)]
  for field in ('staged_output_counts','overshoot_tokens','acceptance_window_means'):
   assert all(x[field]==rankrows[0][field] for x in rankrows)
cleanup=dict(x.split('=',1) for x in (root/'cleanup_status.txt').read_text().splitlines());assert set(cleanup.values())=={'0'}
for kind in ('source','scripts'):assert (root/f'{kind}_before.sha256').read_bytes()==(root/f'{kind}_after.sha256').read_bytes()
out={'arms':arms,'relative_median_tps_percent':100*(arms['cacheon']['median_tps']/arms['cacheoff']['median_tps']-1),'cleanup':cleanup,'input_token_counts_match':True,'all8_cycles_match':True,'limits':'Separate launches and batching may change acceptance trajectory. Residual includes all non-fixed-serving Product work; useful tokens/cycle ratio includes prebulk outputs. No exact generated-text parity claim.'}
(root.parent/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
