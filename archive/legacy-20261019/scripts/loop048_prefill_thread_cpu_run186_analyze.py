#!/usr/bin/env python3
"""Validate legal Run186 Extreme control and thread-CPU phase discrimination."""
import hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'evidence/20260925_loop048_prefill/run186'
bench={k:json.loads((p/f'{k}.json').read_text())['summary'] for k in ['warmup48','control12','measured12']}
for k,n in [('warmup48',48),('control12',12),('measured12',12)]:
 x=bench[k];assert x['n']==x['success']==n and x['fail']==0 and x['max_tokens']==1024 and x['concurrency']==12
ctrl={};detail={}
for rank in range(8):
 c=[json.loads(x) for x in (p/'phase'/f'control_rank{rank}.jsonl').read_text().splitlines()]
 d=[json.loads(x) for x in (p/'phase'/f'rank{rank}.jsonl').read_text().splitlines()]
 assert len(c)==len(d)==9
 assert all(x['rank']==rank for x in c+d) and all(x['mode']=='NONE' for x in d)
 assert all(x['rank']==rank and x['dsa_count']==x['moe_count']==43 and all(len(x[k])==43 for k in ['dsa_layer_wall_us','dsa_layer_thread_cpu_us','moe_layer_wall_us','moe_layer_thread_cpu_us']) for x in d)
 ctrl[rank]=[x for x in c if x['mode']=='NONE'];detail[rank]=d
ctrl_shapes=[x['num_actual_tokens'] for x in ctrl[0]];detail_shapes=[x['num_actual_tokens'] for x in detail[0]]
assert all([x['num_actual_tokens'] for x in ctrl[r]]==ctrl_shapes and [x['num_actual_tokens'] for x in detail[r]]==detail_shapes for r in range(8))
runtime=[]
for rank in range(8):
 for cohort in range(1,7):
  x=json.loads((p/'runtime'/f'rank{rank}_cohort{cohort}.json').read_text())
  assert x['rank']==rank and x['cohort']==cohort and x['pass'] and x['host_mirror_exact'] and x['oracle_target_calls_after_handoff']==0
  runtime.append(x)
restore=json.loads((p/'patch_restore.json').read_text())
sources={'dsa':'/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/dsa.py','moe':'/data/wio/vllm_ascend_26/framework/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py','runner':'/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'}
for k,path in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==restore['files'][k]['restored_sha256']
rank_summary={}
for rank,rr in detail.items():
 def med(k):return statistics.median(x[k] for x in rr)
 rank_summary[str(rank)]={'calls':9,'forward_wall_sum_s':sum(x['forward_wall_ms'] for x in rr)/1000,'forward_thread_cpu_sum_s':sum(x['forward_thread_cpu_ms'] for x in rr)/1000,'forward_wall_minus_thread_cpu_sum_s':sum(x['forward_wall_ms']-x['forward_thread_cpu_ms'] for x in rr)/1000,'forward_thread_cpu_over_wall_median':statistics.median(x['forward_thread_cpu_ms']/x['forward_wall_ms'] for x in rr),'dsa_wall_ms_median':med('dsa_wall_ms'),'dsa_thread_cpu_ms_median':med('dsa_thread_cpu_ms'),'moe_wall_ms_median':med('moe_wall_ms'),'moe_thread_cpu_ms_median':med('moe_thread_cpu_ms'),'control_forward_wall_ms_median':statistics.median(x['forward_wall_ms'] for x in ctrl[rank]),'detailed_forward_wall_ms_median':med('forward_wall_ms')}
latest_wall=sum(max(detail[r][i]['forward_wall_ms'] for r in range(8)) for i in range(9))/1000
latest_cpu=sum(max(detail[r][i]['forward_thread_cpu_ms'] for r in range(8)) for i in range(9))/1000
out={'run':'run186','route':'same-service frozen Extreme warmed48 + control12 + detailed12; legal max_tokens1024','bench':bench,'control_raw_calls_per_rank':9,'control_prefill_calls_per_rank':len(ctrl[0]),'control_shapes':ctrl_shapes,'detailed_shapes':detail_shapes,'rank_summary':rank_summary,'per_call_max_rank_forward_wall_sum_s':latest_wall,'per_call_max_rank_forward_thread_cpu_sum_s':latest_cpu,'runtime_record_count':len(runtime),'source_restored':True,'limits':['Control/detailed cohorts have different scheduler batch shapes and run sequentially; their client throughput and forward medians are not a causal A/B overhead measurement.','Thread CPU close to wall means the measured thread actively executed native/Python submission; it does not prove all time is removable Python overhead.','Host Event::wait wall is not device wait or a safely removable synchronization.','No per-layer device synchronization was added. Diagnostic client envelopes are not formal 48-request E2E.']}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'control_raw_calls_per_rank':9,'control_prefill_calls_per_rank':len(ctrl[0]),'control_shapes':ctrl_shapes,'detailed_shapes':detail_shapes,'latest_forward_wall_s':latest_wall,'latest_forward_thread_cpu_s':latest_cpu,'rank_cpu_wall_fraction_range':[min(x['forward_thread_cpu_over_wall_median'] for x in rank_summary.values()),max(x['forward_thread_cpu_over_wall_median'] for x in rank_summary.values())],'source_restored':True},indent=2))
