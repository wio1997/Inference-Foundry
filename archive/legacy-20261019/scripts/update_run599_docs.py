#!/usr/bin/env python3
"""Record the reviewed Run599 dependency split without a numeric Bound."""
from pathlib import Path

root=Path('/data/wio/Inference_Foundry')
blocks={
'ACHIEVABLE_BOUND.md': '''
## Run599 structural Scheduling split and acceptance support scope

The 12-source/6-evidence pinned read-only Run599 review and independent Astra source audit identify a potential legal **dependency split**, not an overlap saving: after Target forward and its conditional FlashComm output gather, aux hidden plus old Target positions/context slots can feed Draft combine and three-layer context KV; Target logits→greedy acceptance separately supplies raw rejection-dependent query geometry and masked/preserved query seed. Both branches must complete before the Draft query consumer. Python return is not device readiness, and current storage aliases, actual writer/consumer completion and mixed contention remain unproved. Historical R09/R21 and current Run579/580 are mechanism priors, not verdict transfers.

Run597's49,523 active-slot greedy prefix-decision witness positions (including371 internal overshoot) describe only support for observed active output. Formal Run99 staged totals49,470/49,508/49,471 are comparable **non-Bound workload columns** with overshoots318/356/319; neither set gives necessary fresh Target logits/model work, since parked raw acceptance and Draft context/aux/KV consumers remain outside it. Run153/238 already captured the formal11–17s client-minus-decode range, and Run239/241 plus Run341 partially locate it in prefill/seed preparation; a fresh subtraction would duplicate old work. The next primary Product Bound measurement should join actual admission→residual prefill→DSpark seed/KV/state-ready→first Target-ready→Runtime→publication on one fixed W₀; the narrow Target-tail/context timing cut follows if its critical-path sensitivity remains high. Strict finite Resource/Scheduling/Product endpoints and Current→credible-limit gap remain null; Formal Current571.681 tok/s. See Run599 review and PK-097.
''',
'PERFORMANCE_MAP.md': '''
## Run599 dependency frontier and Product boundary priority

Source-typed Target aux→Draft context KV can precede acceptance-dependent Draft query preparation, but no ready/completion or legal mixed-overlap time is measured. The49,523 Run597 active greedy-prefix support positions, and formal Run99 staged totals49,470/49,508/49,471, are semantic support/accounting columns only; they are not compulsory fresh Target work. Existing Run153/238/239/241/341 already account for the large Product boundary remainder and residual prompt microbatches. Next close one same-W₀ request→prefill/seed/KV/state→first Target-ready dependency with all8 timestamps and output lineage before ranking a narrow tail overlap against it. Current571.681 tok/s; finite Bound endpoints null. See PK-097.
''',
'PROJECT_STATE.md': '''
## Run599 Bound-first source gate

Twelve current source files and six evidence inputs pin a conservative Target forward/aux→Draft context versus Target logits/acceptance→Draft query dependency split. Astra scoped review passes source meaning but no device readiness or overlap proof. Run99 formal staged49,470/49,508/49,471 and Run597 active49,523 are non-Bound acceptance support counts, not necessary Target arithmetic. Reuse Run153/238/239/241/341 Product boundary evidence; next close same-W₀ admission→residual prefill→seed/KV/state→first Target-ready, keeping exact-board Resource certificates parallel. Formal Current571.681 tok/s; numeric credible-limit gap unknown.
''',
'RESULTS.md': '''
## Run599 read-only dependency frontier

Source and Astra review support an aux/context branch separate from acceptance/query preparation after the full Target forward/gather producer; no legal device overlap or timing result. Run597 active prefix-decision support49,523 and Run99 staged totals49,470/49,508/49,471 are non-Bound counts. Existing Run153/238/239/241/341 establish the larger unresolved Product preparation path, so no duplicate E2E subtraction was run. Formal Current571.681 tok/s, strict Bound endpoints null. See Run599 findings/review.
'''
}
for name,block in blocks.items():
 p=root/name
 text=p.read_text()
 if '## Run599 ' in text:raise ValueError(name+' already updated')
 p.write_text(text.rstrip()+'\n'+block)
 print(name)
