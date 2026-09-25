#!/usr/bin/env python3
"""Validate legal same-service prefill-only same-stream diagnostic."""
import json,glob,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'evidence/20260925_loop052_prefill_submission/run200'
bench={name:json.loads((out/f'{name}.json').read_text()) for name in ('warmup48','A','B','A2')}
assert all(x['summary']['success']==x['summary']['n'] and x['summary']['max_tokens']==1024 for x in bench.values())
phases={}
for name in ('A','B','A2'):
 rankrows=[]
 for rank in range(8):
  p=out/'phase'/name/f'rank{rank}.jsonl'
  rows=[json.loads(line) for line in p.read_text().splitlines()]
  pre=[x for x in rows if x.get('dsa_count')==43 and x.get('moe_count')==43 and x.get('mode')=='NONE']
  assert pre,(name,rank,len(rows))
  rankrows.append(pre)
 nset={len(x) for x in rankrows}
 shapes=[[x['num_actual_tokens'] for x in r] for r in rankrows]
 assert len(nset)==1 and len(set(tuple(x) for x in shapes))==1,(name,nset,shapes)
 maxsum=sum(max(rankrows[r][i]['forward_wall_ms'] for r in range(8)) for i in range(len(rankrows[0])))/1000
 cpusum=sum(max(rankrows[r][i]['forward_thread_cpu_ms'] for r in range(8)) for i in range(len(rankrows[0])))/1000
 phases[name]={'prefill_calls_per_rank':len(rankrows[0]),'prefill_actual_tokens_by_call':shapes[0],'max_rank_forward_wall_sum_s':maxsum,'max_rank_thread_cpu_sum_s':cpusum,'rank_forward_wall_sum_s':[sum(x['forward_wall_ms'] for x in rr)/1000 for rr in rankrows],'rank_median_dsa_wall_ms':[statistics.median(x['dsa_wall_ms'] for x in rr) for rr in rankrows],'rank_median_moe_wall_ms':[statistics.median(x['moe_wall_ms'] for x in rr) for rr in rankrows]}
runtime={}
for name,cohort in (('A',5),('B',6),('A2',7)):
 rr=[]
 for rank in range(8):
  p=out/'runtime'/f'rank{rank}_cohort{cohort}.json'
  x=json.loads(p.read_text());assert x['pass'] and x['host_mirror_exact'] and x['target_graph_mode']=='FULL' and x['oracle_target_calls_after_handoff']==0 and x['model_runner_cycles_after_handoff']==0
  rr.append(x)
 runtime[name]={'cohort':cohort,'cycles_by_rank':[x['cycles'] for x in rr],'latest_rank_wall_s':max(x['wall_seconds'] for x in rr),'all_rank_pass':True}
hashes={name:[r['output_sha256'] for r in bench[name]['requests']] for name in ('A','B','A2')}
counts={pair:sum(a==b for a,b in zip(hashes[pair[0]],hashes[pair[1]])) for pair in (('A','B'),('A','A2'),('B','A2'))}
marks=glob.glob(str(out/'marks'/'rank*.json'))
result={'run':'run200','contract':'frozen Extreme warmup48 + A12/B12/A2 12, max_tokens1024, same service; B prefill-only shared expert same-stream','bench_summary':{k:v['summary'] for k,v in bench.items()},'phase':phases,'runtime':runtime,'output_hash_equal_request_counts':{f'{a}_{b}':n for (a,b),n in counts.items()},'candidate_rank_marks':sorted(int(Path(p).stem[4:]) for p in marks),'source_roundtrip':{'phase':json.loads((out/'phase_restore.json').read_text()),'candidate':json.loads((out/'candidate_restore.json').read_text())},'decision':'Reject prefill-only same-stream candidate: B and A2 have identical eight-call token shapes, but B max-rank forward sum is18.432ms slower; client B22.551s vs A2 21.330s is confounded by323 vs299 decode cycles. A/A2 output hashes match0/12 even without candidate, so hashes cannot establish exact candidate parity. No formal E2E.','limits':['Different cohort batching/shapes and order can confound local timing and acceptance; use A/A2 controls.','Output hash at temperature0 is a client response check, not internal exact tensor/state parity.','Diagnostic12-request cohorts are not formal repeated48-request E2E.']}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'bench_duration_s':{k:v['summary']['duration_s'] for k,v in bench.items()},'phase':{k:{x:v[x] for x in ('prefill_calls_per_rank','max_rank_forward_wall_sum_s','prefill_actual_tokens_by_call')} for k,v in phases.items()},'runtime':runtime,'hash_equal':result['output_hash_equal_request_counts'],'marks':result['candidate_rank_marks']},indent=2))
