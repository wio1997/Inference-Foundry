from __future__ import annotations
import json
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
s=json.loads((ROOT/'evidence/20260928_loop081_bound/run592/summary.json').read_text())
assert s['status']=='historical_instrumented_current_slice_only' and len(s['rows'])==16
assert s['metrics_us']['partial_start_to_copy_end']['median']==31.7695
assert all(s[x] is None for x in ('strict_resource_bound_s','strict_scheduling_bound_s','strict_product_bound_tps'))
notes={
'ACHIEVABLE_BOUND.md':'''## Run592 reused Level1 historical native timing — Current only

An offline reducer and independent Astra High review recovered two native Model45 partial MatMul→BF16 RS→SDMA copy→HC-post neighborhoods per rank from Run246's existing all8 production Level1 MemoryAccess traces. Across16 rank-local occurrences, partial start→copy end is30.320–33.441µs (median31.7695µs); partial end→copy end15.780–18.440µs (median17.32975µs); copy end→following HC-post start0.020–0.0405µs. Thirty-two raw SHA, task_time cross-checks, session ordering and four negative mutations pass. This calibrates a **historical instrumented Current DAG cost** and makes a large idle gap at that profiled adjacency unlikely. The profile lacks direct cohort/request ID, comes from a different Graph generation than Run589, and has Level1/synchronization perturbation. It cannot be multiplied by layers/cycles, equated to unmarked Current, apportioned into HCCL transit/peer wait, or promoted to removable time. A duplicate live Level0 slice has low immediate value; prioritize broad fixed-`W₀` logical work/traffic and whole Target/Draft/KV/Host all8 dependency evidence while exact-board `C⁺/B` remains open. Formal Current571.681 tok/s; strict finite Resource/Scheduling/Product endpoints and numerical Current→credible-limit gap remain null. See Run592 findings/review and PK-090.
''',
'PERFORMANCE_MAP.md':'''## Run592 historical native timing calibration

Existing Run246 Level1 trace supplies a rank-local Model45 partial→RS→SDMA copy historical instrumented interval:31.7695µs median start-to-copy-end over16 all8 occurrences. The copy→HC-post start gap is0.020–0.0405µs in that trace. This eliminates a speculative large idle-gap hypothesis for one profiled adjacency, but missing direct cohort/request join, cross-run semantic identity and profiler perturbation prevent unmarked Current or Bound promotion. A fresh narrow profile is deferred until it changes a decision; broader fixed-work and full critical-path certificates lead. Current571.681 tok/s; strict numeric endpoints remain null. See PK-090.
''',
'PROJECT_STATE.md':'''## Run592 reused native time checkpoint

Run246 existing Level1 traces were reduced offline across8 ranks×2 repeats and independently reviewed. Historical instrumented partial-start→copy-end median31.7695µs, no direct profile→cohort/request join, no same-run Run589 identity, no unmarked timing or Bound. Do not launch a duplicate live narrow timing run without a decision-relevant reason. Next broad fixed-`W₀` compulsory-work/traffic and full Target/Draft/KV/Host DAG, with exact-board capacity work in parallel. Formal Current571.681 tok/s; numerical credible-limit gap unknown.
''',
'RESULTS.md':'''## Run592 offline Run246 native slice time

All8×2 existing Level1 occurrences pass exact Model/physical-stream/task selection, trace/task_time corroboration,32 SHA and Astra review. Partial start→copy end median31.7695µs; copy→HC-post start0.020–0.0405µs. Historical instrumented Current only; no fresh NPU run, formal E2E gain or finite Bound. See Run592 findings.
'''}
for name,body in notes.items():
 p=ROOT/name;text=p.read_text();assert body.splitlines()[0] not in text;p.write_text(text.rstrip()+'\n\n'+body)
entry={'id':'PK-090',
 'topic':'Existing production Level1 trace can time a native partial→RS→copy cut, but one narrow slice is not a Product gap',
 'mechanism':'Model/physical-stream/task plus occurrence chronology in Run246 trace_view identifies two native neighborhoods/rank; task_time independently corroborates the SDMA and other role intervals within the same profiler export.',
 'environment':'Run246 all8 DP1TP8 FULL Graph fixed DSpark7 diagnostic, CANN9.1 Level1 MemoryAccess and target synchronization; offline Run592 reducer and Astra review; Run589 semantic tags are from another acquisition.',
 'observed':'Sixteen rank-local occurrences give partial-start→SDMA-copy-end30.320–33.441us, median31.7695us; partial-end→copy-end median17.32975us; copy→HC-post start0.020–0.0405us. All32 source SHA, four negative mutations and independent raw recomputation pass.',
 'failure_or_limit':'No direct profile→cohort/request ID, same-generation Run589 semantic join, matched unmarked timing or fixed-Run99 W0; Level1/sync perturbs schedule. Do not multiply one slice by layers/cycles, isolate HCCL transit from peer wait or infer removable Product wall-time.',
 'revalidate_when':'A current same-generation native timing or lower-perturbation acquisition becomes necessary for a concrete broad-DAG/overlap decision, with all8 workload ledger and matched no-marker control; otherwise reuse historical interval and pursue larger Bound uncertainties.',
 'extreme_relation':'Narrows conditional Current Scheduling node cost and rejects a large idle gap at one historical profiled adjacency; strict finite Resource/Scheduling/Product endpoints and numeric Current-to-limit gap stay null, Formal Current571.681tok/s.',
 'source':[{'repository':'Inference_Foundry','ref':'run592','path':'evidence/20260928_loop081_bound/run592/summary.json'},
           {'repository':'Inference_Foundry','ref':'run592','path':'evidence/20260928_loop081_bound/run592/findings.md'},
           {'repository':'Inference_Foundry','ref':'run592','path':'evidence/20260928_loop081_bound/run592/astra_final_review.md'}],
 'status':'current_diagnostic'}
p=ROOT/'performance_knowledge/entries.jsonl';ids={json.loads(x)['id'] for x in p.read_text().splitlines() if x.strip()};assert entry['id'] not in ids
with p.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
print('Run592 docs and PK appended')
