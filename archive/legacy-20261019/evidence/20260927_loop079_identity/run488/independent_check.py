import sys,json,hashlib,statistics,math,re
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/data/wio/Inference_Foundry')
sys.path.insert(0,str(ROOT/'scripts'))
import loop079_target_frontier_validate_run478 as val
import loop079_target_frontier_reduce_run487 as reducer
import loop079_target_frontier_patch_run478 as patcher
out=ROOT/'evidence/20260927_loop079_identity/run488';out.mkdir(parents=True,exist_ok=True)
run=ROOT/'evidence/20260927_loop079_identity/run484/b_candidate'
saved=json.loads((ROOT/'evidence/20260927_loop079_identity/run487/intervals.json').read_text())
local=val.validate_root(run);final=val.final_admit(run)
assert local==json.loads((run/'validation.json').read_text())
assert final==json.loads((run/'final_admission.json').read_text())
recomputed=reducer.reduce(run)
assert (ROOT/saved['source_root']).resolve()==run.resolve()
recomputed['source_root']=saved['source_root']
assert recomputed==saved
for name,digest in saved['input_sha256'].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
pins=json.loads((ROOT/'evidence/20260927_loop079_identity/run486/checks_final.json').read_text())['reviewed_hashes']
for name,digest in pins.items():assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()==digest
source_status=[]
for row in val.SOURCE_MANIFEST:
 source=Path(row['source'])
 assert hashlib.sha256(source.read_bytes()).hexdigest()==row['original']
 generated=patcher.patch(row['key'],source.read_text())
 assert hashlib.sha256(generated.encode()).hexdigest()==row['patched']
 compile(generated,str(source),'exec')
 source_status.append(dict(path=str(source),sha256=row['original']))
for name,digest in json.loads((ROOT/'evidence/20260927_loop079_identity/run469/source_hashes.json').read_text()).items():
 if name.startswith('source/'):assert hashlib.sha256((ROOT/'evidence/20260927_loop079_identity/run469'/name).read_bytes()).hexdigest()==digest
rows=[json.loads(p.read_text()) for p in sorted((run/'capture').glob('*.json'))]
runtime={(r['rank'],r['cohort']):r for p in (run/'runtime').glob('*.json') for r in [json.loads(p.read_text())]}
assert len(rows)==len(runtime)==40
assert {(r['rank'],r['cohort']) for r in rows}=={(r,c) for r in range(8) for c in range(1,6)}
def stats(v):
 q=statistics.quantiles(v,n=4,method='inclusive');med=statistics.median(v)
 return dict(n=len(v),min=min(v),q1=q[0],median=med,q3=q[2],max=max(v),mad=statistics.median(abs(x-med) for x in v))
edgelist=reducer.EDGES
intervals={edge:stats([r['intervals'][edge] for r in rows]) for edge in edgelist}
intervals['T_U_adjacent_ms']=stats([sum(r['intervals'].values()) for r in rows])
for edge,a in intervals.items():
 s=saved['intervals_ms'][edge]
 assert s['n']==a['n']
 for stat in ['min','median','max']:assert abs(s[stat+'_ms']-a[stat])<1e-12
cohorts=[]
for c in range(1,6):
 cr=[r for r in rows if r['cohort']==c];r=cr[0]
 counts=r['accepted_counts'];staged=[sum(v[i] for v in counts) for i in range(12)]
 useful=sum(min(n,1024) for n in staged);ov=sum(staged)-useful
 assert useful==12288
 for rr in cr:
  rt=runtime[rr['rank'],c]
  assert rr['accepted_counts']==counts and rr['initial_output_counts']==[0]*12
  assert rt['handoff_before_model_forward'] and rt['target_graph_requested'] and rt['model_runner_cycles_after_handoff']==0
  assert rt['staged_output_counts']==staged and rt['overshoot_tokens']==ov
 assert saved['cohorts'][str(c)]['cycles']==r['cycles']
 assert saved['cohorts'][str(c)]['staged_tokens']==sum(staged)
 assert saved['cohorts'][str(c)]['overshoot_tokens']==ov
 cs={e:stats([x['intervals'][e] for x in cr]) for e in edgelist}
 for e,s in cs.items():
  for st in ['min','median','max']:assert abs(s[st]-saved['cohorts'][str(c)]['intervals_ms'][e][st+'_ms'])<1e-12
 cohorts.append(dict(cohort=c,cycles=r['cycles'],useful=useful,staged=sum(staged),overshoot=ov,selected_cycle64_accepted=sum(counts[64]),intervals=cs))
for r in rows:
 assert list(r['markers'])==['T','R0','R1','H','U']
 assert len(r['native_calls'])==4
 assert r['replay']['capture_scope']['origin']=='startup_full_decode96'
 assert r['graph_update_branch']=='after'
 assert r['graph_update'][0]['private_stream_relation']=='unresolved'
 assert r['stream'][1]==dict(stream_id=0,device_index=r['rank'],device_type=20)
 assert all(math.isfinite(v) and v>=0 for v in r['intervals'].values())
 assert r['replay']['selected_observation_ordinal']==r['cohort']
 assert len({(x['replay']['entry_id'],x['replay']['graph_id'],x['replay']['capture_generation']) for x in rows if x['rank']==r['rank']})==1
sync=stats([(r['sync']['end_ns']-r['sync']['begin_ns'])/1e6 for r in rows])
host_t_r0=stats([(r['markers']['R0']['host_ns']-r['markers']['T']['host_ns'])/1e6 for r in rows])
logpath=Path((run/'server_log_path.txt').read_text().strip());log=logpath.read_text(errors='replace')
posts=[line for line in log.splitlines() if 'POST /v1/chat/completions' in line]
assert len(posts)==60 and all('200 OK' in line for line in posts)
bad=[line for line in log.splitlines() if re.search(r'\bERROR\b|Traceback|OUT_OF_SCOPE|RuntimeError|AssertionError',line)]
assert not bad
warnings=[line for line in log.splitlines() if 'WARNING' in line]
clients=[json.loads((run/n).read_text()) for n in ['warmup48.json','bench.json']]
requests=[x for c in clients for x in c['requests']]
events=sorted([(x['start'],1) for x in requests]+[(x['end'],-1) for x in requests])
active=peak=0
for _,d in events:active+=d;peak=max(peak,active)
assert active==0 and peak==12
assert max(x['end'] for x in clients[0]['requests'])<=min(x['start'] for x in clients[1]['requests'])
cleanup=dict(x.split('=',1) for x in (run/'cleanup_status.txt').read_text().splitlines())
assert len(cleanup)==8 and set(cleanup.values())=={'0'}
assert json.loads((run/'stop_process_probe.txt').read_text())==[]
assert json.loads((run/'stop_orphan_bench.txt').read_text())==[]
npu=(run/'stop_npu_probe.txt').read_text()
assert npu.count('No running processes found in NPU')==8
hbm=[int(x) for x in re.findall(r'(\d+) / 65536',npu)]
assert len(hbm)==8 and max(hbm)<5000
result=dict(status='PASS_SCOPED_DIAGNOSTIC',validation=local,final_admission=final,hashed_input_files=len(saved['input_sha256']),reviewed_script_hashes=pins,live_sources=source_status,raw_interval_recompute='all overall and per-cohort min/median/max match 1e-12ms',intervals_ms=intervals,cohorts=cohorts,sync_host_ms=sync,T_R0_host_ms=host_t_r0,HTTP200_posts=len(posts),client_peak_concurrency=peak,warning_line_count=len(warnings),error_lines=bad,input_token_range=[min(x['input_tokens'] for x in requests),max(x['input_tokens'] for x in requests)],cleanup=cleanup,saved_stop_HBM_MiB=hbm,server_log_sha256=hashlib.sha256(logpath.read_bytes()).hexdigest(),reducer_sha256=hashlib.sha256((ROOT/'scripts/loop079_target_frontier_reduce_run487.py').read_bytes()).hexdigest())
(out/'independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(out/'server_warnings.txt').write_text('\n'.join(warnings)+'\n')
print(json.dumps({k:result[k] for k in ['status','hashed_input_files','intervals_ms','sync_host_ms','T_R0_host_ms','HTTP200_posts','client_peak_concurrency','warning_line_count','input_token_range','saved_stop_HBM_MiB']},indent=2))
