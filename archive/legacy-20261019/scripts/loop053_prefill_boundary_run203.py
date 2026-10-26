#!/usr/bin/env python3
"""Map low-overhead prefill layer cost to static DSA class and profiled call family."""
import json,statistics
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parents[1]
phase=root/'evidence/20260925_loop048_prefill/run186/phase'
classrows=json.loads((root/'evidence/20260925_loop044_target/run151/dsa_chain_audit.json').read_text())['rows']
cls={x['layer']:x['class'] for x in classrows if x['rank']==0 and x['cycle']==64}
assert len(cls)==43
byclass=defaultdict(lambda:{'dsa_us':[],'moe_us':[]})
for rank in range(8):
 rows=[json.loads(x) for x in (phase/f'rank{rank}.jsonl').read_text().splitlines()]
 assert len(rows)==9
 for row in rows:
  assert row['dsa_count']==43 and row['moe_count']==43 and row['mode']=='NONE'
  for i,c in cls.items():
   byclass[c]['dsa_us'].append(row['dsa_layer_wall_us'][i]);byclass[c]['moe_us'].append(row['moe_layer_wall_us'][i])
classes={c:{'layers':sum(v==c for v in cls.values()),'dsa_wall_us_median':statistics.median(x['dsa_us']),'moe_wall_us_median':statistics.median(x['moe_us']),'samples':len(x['dsa_us'])} for c,x in byclass.items()}
prof=json.loads((root/'evidence/20260925_loop048_prefill/run181/analysis.json').read_text())['major_scope_summary']
family={k:{'scope_count':v['scope_count'],'inclusive_total_ms':v['inclusive_total_ms'],'direct_child_uncovered_total_ms':v['direct_child_uncovered_total_ms'],'top_direct_children':v['top_direct_children'][:18]} for k,v in prof.items() if k in ('vllm::dsa_forward','vllm::moe_forward_shared')}
result={'run':'run203','low_overhead_source':'Run186 eight-rank nine-call legal Extreme measured cohort','profile_source':'Run181 rank0 Run165 one83-token profiler-perturbed forward','classes':classes,'profiled_call_families':family,'model_boundary_findings':['43 DSA+43 MoE Python custom-op bodies are repeated in each eager prefill forward.','DSA body resolves layer and metadata, builds KV tuple, then calls borrowed impl.forward; MoE body resolves layer then invokes _forward_impl. Both then submit many required torch/custom/HCCL operators.','Run181 direct-child-uncovered DSA80.1ms and MoE82.9ms include Python, native context, profiler callbacks and runtime work. They cannot be credited to a native rewrite.','Run186 low-overhead per-layer scopes provide the actual gross Host envelope; dynamic token count and request count vary across calls, so a single static prefill graph does not cover the cohort without a replay-input/state ABI.'],'limits':['Run151 layer class is from separate target decode profile, but model layer compress-ratio configuration is static; per-layer mapping is reusable for attribution.','Run181 and Run186 are separate services and different instrumentation; no subtractive estimate is made.','Neither inclusive Host duration nor direct-child-uncovered duration is a removable-time bound.']}
out=root/'evidence/20260925_loop053_native_prefill/run203/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(classes,indent=2))
