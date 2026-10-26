from __future__ import annotations
import json
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
notes={
'ACHIEVABLE_BOUND.md':'''## Run591 source-typed attention consumer edge

Astra High's read-only Run591 review pins the installed DSA wrapper, DeepSeek V4 decoder layer, C++ schema/adapter and HC-post package. The attention destination is passed through storage-preserving `view` and `unsqueeze` to HC-post's typed `x` input; the source has no extra activation materialization on this edge. Run589 all8 same-run native words corroborate the relation, but do not certify the actual compiled ACL tensor descriptor, exported argument-word ABI, all cross-stream writers or allocation lifetime. The source-level edge helps define the production critical-path cut. A bounded Level0 acquisition can first measure the already identified partial→RS→copy *instrumented Current* interval; unmarked Current, removable time or lower Bound requires perturbation controls, fixed-`W₀` state/ledger and rank-local completion evidence. Do not multiply one layer0 interval by43. Formal Current571.681 tok/s; strict finite Resource/Scheduling/Product endpoints and numeric gap remain null. See Run591 findings and PK-089.
''',
'PERFORMANCE_MAP.md':'''## Run591 consumer dependency qualification

Pinned source establishes an alias-only attention output→HC-post typed-input read edge, and Run589 all8 native argument words corroborate it. Compiled ACL descriptor, full last-writer census and timing remain open. The immediate Scheduling measurement target is the narrower partial→RS→copy cut with a validated Level0 time source and same-generation task correlation; causal savings still require unmarked/fixed-`W₀` controls and formal E2E. No finite Bound or numerical Current→limit gap yet. See PK-089.
''',
'PROJECT_STATE.md':'''## Run591 Scheduling source review

Read-only Astra High review PASS for a conditional source-typed attention output→HC-post x storage/read dependency, corroborated by Run589 all8 native words. It does not certify actual compiled ABI, full allocation lifetime or time. Proceed to bounded rank-local timing of the partial→RS→copy cut without waiting for a full HC-post native certificate; keep Resource W⁻/C⁺ work parallel. Formal Current571.681 tok/s and strict Bound endpoints remain unknown.
''',
'RESULTS.md':'''## Run591 source-typed consumer review

Pinned DSA/model/C++/adapter/package source supports alias-only attention destination→HC-post typed x read, with all8 Run589 native word corroboration. No compiled ACL descriptor, exhaustive last writer, runtime interval or finite Bound. See Run591 findings.
'''}
for name,body in notes.items():
 p=ROOT/name;s=p.read_text();assert body.splitlines()[0] not in s;p.write_text(s.rstrip()+'\n\n'+body)
entry={'id':'PK-089','topic':'DeepSeek V4 attention copy output is a source-typed alias-only HC-post input, with native ABI still open',
 'mechanism':'DSA custom op mutates caller output; wrapper view and decoder HC-post unsqueeze preserve storage. C++ typed x input is passed unchanged through ACLNN adapter; the source has no intermediate activation copy.',
 'environment':'Run591 pinned installed DeepSeek V4/DSA/C++ source and packaged HC-post object on 8×910B3; Run589 same-run measured FULL96 Graph under fixed DSpark7.',
 'observed':'All8 Run589 HcPost task44 word0 equals labeled attention destination and next HcPre shows plausible post-consumption pool reuse; pinned source and schema establish the conditional typed read edge. Source_review.json hashes nine source files and 24 Graph/capture artifacts.',
 'failure_or_limit':'Native argument word roles, actual compiled ACL tensor descriptor, all cross-stream writers and per-cycle allocation lifetime are not certified. No task timing or shorter legal schedule; repeated pointer addresses cannot be joined across pool lifetimes.',
 'revalidate_when':'A rank-local same-generation Level0 production partial→RS→copy time acquisition with validated task identity and matched marker/profiler perturbation control; full HC-post native consumer claims require typed ACL dispatch/descriptor evidence.',
 'extreme_relation':'Narrows source dependency for Scheduling DAG without changing fixed algorithm, Formal Current571.681tok/s, strict finite Bound endpoints or numeric Current-to-limit distance.',
 'source':[{'repository':'Inference_Foundry','ref':'run591','path':'evidence/20260928_loop081_bound/run591/source_review.json'},
           {'repository':'Inference_Foundry','ref':'run591','path':'evidence/20260928_loop081_bound/run591/findings.md'}],
 'status':'conditional_source_contract'}
p=ROOT/'performance_knowledge/entries.jsonl';ids={json.loads(x)['id'] for x in p.read_text().splitlines() if x.strip()};assert entry['id'] not in ids
with p.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
print('Run591 docs and PK appended')
