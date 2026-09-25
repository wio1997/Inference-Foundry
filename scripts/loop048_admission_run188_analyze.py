#!/usr/bin/env python3
"""Validate Run188 legal same-service admission A/B/A-prime diagnostic."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'evidence/20260925_loop048_prefill/run188'
bench={tag:json.loads((p/f'{tag}.json').read_text())['summary'] for tag in ['warmup48','A','B','A2']}
for tag,n in [('warmup48',48),('A',12),('B',12),('A2',12)]:
 x=bench[tag];assert x['n']==x['success']==n and x['fail']==0 and x['max_tokens']==1024 and x['concurrency']==12
wait=[json.loads(x) for x in (p/'wait.jsonl').read_text().splitlines()]
assert len(wait)==1 and wait[0]['counts_before']==[0,1] and sum(wait[0]['counts_after'])==12 and 0<wait[0]['wait_s']<=2.02
rows={};runtime={}
for tag,cohort in [('A',5),('B',6),('A2',7)]:
 rows[tag]={};runtime[tag]={}
 for rank in range(8):
  r=[json.loads(x) for x in (p/'forward'/f'rank{rank}_{tag}.jsonl').read_text().splitlines()]
  assert all(x['rank']==rank and x['tag']==tag for x in r)
  rows[tag][rank]=r
  x=json.loads((p/'runtime'/f'rank{rank}_cohort{cohort}.json').read_text())
  assert x['rank']==rank and x['cohort']==cohort and x['pass'] and x['host_mirror_exact'] and x['oracle_target_calls_after_handoff']==0 and x['generated_output_counts']==[1024]*12
  runtime[tag][rank]=x
for cohort in range(1,5):
 for rank in range(8):
  x=json.loads((p/'runtime'/f'rank{rank}_cohort{cohort}.json').read_text())
  assert x['pass'] and x['host_mirror_exact']
assert len(list((p/'runtime').glob('rank*_cohort*.json')))==56
for tag in ['A','B','A2']:
 shapes=[[x['num_actual_tokens'] for x in rows[tag][rank] if x['mode']=='NONE'] for rank in range(8)]
 assert all(x==shapes[0] for x in shapes)
summary={}
for tag in ['A','B','A2']:
 prefill=[[x for x in rows[tag][rank] if x['mode']=='NONE'] for rank in range(8)]
 count={len(x) for x in prefill};assert len(count)==1
 summary[tag]={'client_envelope_s':bench[tag]['duration_s'],'client_output_tps_diagnostic':bench[tag]['output_tps'],'ttft_p50_ms':bench[tag]['ttft_ms_p50'],'prefill_calls':next(iter(count)),'prefill_tokens':[x['num_actual_tokens'] for x in prefill[0]],'max_rank_prefill_forward_wall_sum_s':sum(max(prefill[rank][i]['forward_wall_ms'] for rank in range(8)) for i in range(len(prefill[0])))/1000,'decode_cycles':max(runtime[tag][rank]['cycles'] for rank in range(8)),'latest_rank_runtime_wall_s':max(runtime[tag][rank]['wall_seconds'] for rank in range(8))}
restore=json.loads((p/'patch_restore.json').read_text())
sources={'core':'/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/core.py','runner':'/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'}
for k,path in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==restore['files'][k]['restored_sha256']
control_mean=(summary['A']['client_envelope_s']+summary['A2']['client_envelope_s'])/2
out={'run':'run188','route':'same-service legal warmed Extreme no-wait A / <=2s Core first-cohort wait B / no-wait A-prime','bench':bench,'wait':wait,'phase_summary':summary,'B_client_envelope_delta_vs_control_mean_s':summary['B']['client_envelope_s']-control_mean,'runtime_record_count':56,'sources_restored':True,'decision':'REJECT 2s initial admission wait as a product candidate: exact full-cohort consolidation occurred, but only a 0.163s diagnostic client-envelope improvement versus the two-control mean, not material or formal; B also had 32 more decode cycles and longer Runtime wall. No formal E2E warranted.','limits':['Single service, one B cohort and sequential A/B/A-prime; decode acceptance/route variation and cache state may differ. Do not causally attribute all 32 extra cycles to waiting.','Client envelope differences of 0.05-0.27s are smaller than prior diagnostic variability and do not prove product gain.','Requests passed length/Runtime gates, not independent full semantic equivalence.','Local prefill wall and Core wait are not additive end-to-end savings; scheduler and device work overlap.']}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'wait':wait,'phase_summary':summary,'B_delta_vs_controls_s':out['B_client_envelope_delta_vs_control_mean_s'],'sources_restored':True},indent=2))
