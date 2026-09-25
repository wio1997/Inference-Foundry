#!/usr/bin/env python3
"""Validate Run184 legal Extreme phase timing and source restoration."""
import hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'evidence/20260925_loop048_prefill/run184'
summary={k:json.loads((p/f'{k}.json').read_text())['summary'] for k in ['warmup48','measured12']}
assert summary['warmup48']['n']==summary['warmup48']['success']==48 and summary['warmup48']['fail']==0
assert summary['measured12']['n']==summary['measured12']['success']==12 and summary['measured12']['fail']==0
assert all(x['max_tokens']==1024 and x['concurrency']==12 for x in summary.values())
rows={}
for rank in range(8):
 r=[json.loads(x) for x in (p/'phase'/f'rank{rank}.jsonl').read_text().splitlines()]
 assert len(r)==5 and all(x['rank']==rank and x['dsa_count']==43 and x['moe_count']==43 and x['mode']=='NONE' for x in r)
 rows[rank]=r
shapes=[x['num_actual_tokens'] for x in rows[0]]
assert all([x['num_actual_tokens'] for x in rows[r]]==shapes for r in range(8))
assert len(list((p/'runtime').glob('rank*_cohort*.json')))==40
runtime=[]
for rank in range(8):
 for cohort in range(1,6):
  x=json.loads((p/'runtime'/f'rank{rank}_cohort{cohort}.json').read_text())
  assert x['rank']==rank and x['cohort']==cohort and x['pass'] and x['host_mirror_exact'] and x['oracle_target_calls_after_handoff']==0
  runtime.append(x)
source={'dsa':'/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/dsa.py','moe':'/data/wio/vllm_ascend_26/framework/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py','runner':'/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'}
restore=json.loads((p/'patch_restore.json').read_text())
for k,path in source.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==restore['files'][k]['restored_sha256']
rank_summary={}
for rank,r in rows.items():
 rank_summary[str(rank)]={'calls':5,'forward_sum_s':sum(x['forward_wall_ms'] for x in r)/1000,'dsa_sum_s':sum(x['dsa_wall_ms'] for x in r)/1000,'moe_sum_s':sum(x['moe_wall_ms'] for x in r)/1000,'median_forward_ms':statistics.median(x['forward_wall_ms'] for x in r),'median_dsa_ms':statistics.median(x['dsa_wall_ms'] for x in r),'median_moe_ms':statistics.median(x['moe_wall_ms'] for x in r),'median_dsa_moe_fraction':statistics.median((x['dsa_wall_ms']+x['moe_wall_ms'])/x['forward_wall_ms'] for x in r)}
latest=[max(rows[rank][i]['forward_wall_ms'] for rank in range(8)) for i in range(5)]
latest_runtime=max(x['wall_seconds'] for x in runtime if x['cohort']==5)
out={'run':'run184','route':'Extreme Runtime with static metadata, target graph and DSpark slot refresh enabled','contract':'legal warmed 48+12 requests, c12 1024-token output, 8x910B3 DP1TP8','request_summary':summary,'prefill_actual_tokens_by_call':shapes,'rank_summary':rank_summary,'per_call_max_rank_forward_ms':latest,'sum_per_call_max_rank_forward_s':sum(latest)/1000,'latest_rank_measured_runtime_wall_s':latest_runtime,'runtime_record_count':len(runtime),'source_restored':True,'limits':['No torch profiler; wall timing is low-overhead but wrapper/custom-op timing still perturbs execution.','DSA and MoE wall encompasses Python, op enqueue and waits needed for exact state; scope coverage is not removable time.','Sum of per-call rank maxima is a conservative phase exposure descriptor, not a strict end-to-end critical path or achievable saving.','Diagnostic 12-request client envelope is not formal 48-request E2E; Run183 is a separate-service Stock reference, not an A/B speedup.']}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'shapes':shapes,'latest_forward_s':out['sum_per_call_max_rank_forward_s'],'latest_runtime_wall_s':latest_runtime,'rank_phase_fraction_range':[min(v['median_dsa_moe_fraction'] for v in rank_summary.values()),max(v['median_dsa_moe_fraction'] for v in rank_summary.values())],'source_restored':True},indent=2))
