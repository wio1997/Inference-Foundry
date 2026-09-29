"""Admit shared-service OFF/ON/OFF, preserving global raw cohort identities."""
import hashlib,json,statistics
from pathlib import Path
root=Path('evidence/20260929_loop081_bound/run675/live')
arms={}
for phase,offset in [('off_a',0),('on',16),('off_b',32)]:
 p=root/phase;summary=json.loads((p/'summary.json').read_text());assert summary['pass'] and summary['offset']==offset
 for name in ('warmup','bench48_1','bench48_2','bench48_3'):
  b=json.loads((p/(name+'.json')).read_text());s=b['summary'];r=b['requests']
  assert s['n']==s['success']==len(r)==48 and s['fail']==0 and s['max_tokens']==1024 and s['concurrency']==12
  assert all(x['output_tokens']==1024 and x['error'] is None for x in r)
  ref=json.loads((root/'off_a'/(name+'.json')).read_text())['requests']
  assert {x['i']:x['input_tokens'] for x in r}=={x['i']:x['input_tokens'] for x in ref}
 seen={}
 for f in (p/'runtime').glob('rank*_cohort*.json'):
  r=json.loads(f.read_text());k=(r['rank'],r['cohort']);assert k not in seen;seen[k]=r
  assert r['metadata_graph_mode']=={'phase':phase,'enabled':phase=='on','version':offset//16+1}
  assert r['metadata_graph_capture']==(phase=='on') and r['metadata_graph_replays']==(r['cycles'] if phase=='on' else 0)
  assert r['pass'] and r['host_mirror_exact'] and 'FULL' in r['target_graph_mode'] and r['oracle_target_calls_after_handoff']==0 and r['generated_output_counts']==[1024]*12
 assert set(seen)=={(r,c) for r in range(8) for c in range(offset+1,offset+17)}
 for c in range(offset+1,offset+17):
  for field in ('cycles','staged_output_counts','overshoot_tokens','acceptance_window_means'):
   assert all(seen[r,c][field]==seen[0,c][field] for r in range(8))
 rows=[]
 for rep in range(1,4):
  s=json.loads((p/f'bench48_{rep}.json').read_text())['summary']
  rr=[seen[0,c] for c in range(offset+rep*4+1,offset+rep*4+5)]
  cycles=sum(x['cycles'] for x in rr);wall=sum(x['wall_seconds'] for x in rr)
  rows.append(dict(repeat=rep,output_tps=s['output_tps'],client_s=s['duration_s'],cycles=cycles,runtime_s=wall,runtime_ms_per_cycle=wall/cycles*1000,client_minus_runtime_s=s['duration_s']-wall,product_tokens_per_runtime_cycle=49152/cycles,staged_accepted_tokens_per_cycle=sum(sum(x['staged_output_counts']) for x in rr)/cycles,cohort_cycles=[x['cycles'] for x in rr],cohort_acceptance=[x['acceptance_window_means'] for x in rr]))
 route=summary['routing'];version=offset//16+1
 assert [r['initial_requests'] for r in route]==list(range(192*(version-1)+48,192*version+1,48))
 assert all(r['phase']=='on' and r['enabled'] and r['version']==1 and r['memo_misses']==48 and r['memo_hits']==r['initial_requests']-48 for r in route)
 arms[phase]={'rows':rows,'median_tps':statistics.median(r['output_tps'] for r in rows),'routing':route}
assert len(list((root/'runtime').glob('rank*_cohort*.json')))==384
cleanup=dict(x.split('=',1) for x in (root/'cleanup_status.txt').read_text().splitlines());assert set(cleanup.values())=={'0'}
for kind in ('source','scripts'):
 assert (root/(kind+'_before.sha256')).read_bytes()==(root/(kind+'_after.sha256')).read_bytes()
 for line in (root/(kind+'_after.sha256')).read_text().splitlines():
  sha,path=line.split(None,1);assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
out={'protocol':'single service metadata Graph OFF_A / ON / OFF_B; each warm48+3formal48; input-token and initial-hash caches ON throughout; capture included in Product wall','arms':arms,'on_vs_off_a_percent':100*(arms['on']['median_tps']/arms['off_a']['median_tps']-1),'on_vs_off_b_percent':100*(arms['on']['median_tps']/arms['off_b']['median_tps']-1),'cleanup':cleanup,'limits':'No fixed trajectory assumption. CPU hash time is not independently additive Product savings. Numerical Framework bound remains unknown.'}
(root.parent/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
