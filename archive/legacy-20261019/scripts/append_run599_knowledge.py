#!/usr/bin/env python3
"""Append scoped structural DAG prior and keep numerical Bounds null."""
import json
from pathlib import Path

path=Path('/data/wio/Inference_Foundry/performance_knowledge/entries.jsonl')
entries=[json.loads(x) for x in path.read_text().splitlines() if x]
assert entries[-1]['id']=='PK-096'
entry=dict(id='PK-097',
    topic='Target aux to Draft context can branch before acceptance-dependent query, but execution overlap is unmeasured',
    mechanism='Source-typed split after Target forward plus conditional FlashComm gather: aux and old positions/slots feed Draft combine/context KV, while logits/greedy acceptance feeds raw rejected-count query geometry and masked-or-preserved seed; both join before Draft query.',
    environment='Current DeepSeek V4 Flash W4A8, 8×910B3 DP1TP8 fixed DSpark7; Run599 read-only source audit with Run597/598 new-W0 witness.',
    observed='12 source and6 evidence SHA pass, Astra scoped review of actual call/data flow. Run597 active greedy prefix support49523, formal Run99 staged totals49470/49508/49471 remain non-Bound workload columns.',
    failure_or_limit='Python return is not device ready; actual async writer/consumer, storage lifetime, HCCL/memory competition and perturbation-controlled overlap are not measured. Prefix support excludes parked raw decisions and Draft context; not compulsory fresh Target work.',
    revalidate_when='First join full same-W0 Product admission→residual prefill→seed/KV/state→first Target-ready path; if this cut remains high sensitivity, capture all8 three-cycle producer/logits/acceptance/context/query device events and no-probe control before implementing overlap.',
    extreme_relation='Narrows structural Scheduling uncertainty but leaves Formal Current571.681 tok/s and strict Resource/Scheduling/Product endpoints null.',
    source=[dict(repository='Inference_Foundry',ref='run599',path='evidence/20260928_loop081_bound/run599/source_gate.json'),
            dict(repository='Inference_Foundry',ref='run599',path='evidence/20260928_loop081_bound/run599/astra_review.md'),
            dict(repository='dsv4f-w4a8-ascend-logs',ref='db3beef223e0b5acd81ccccb444600c1b22aac8a',path='rounds/R09_draft_forward_decomposition/REPORT.md'),
            dict(repository='dsv4f-w4a8-ascend-logs',ref='db3beef223e0b5acd81ccccb444600c1b22aac8a',path='rounds/R21_dsa_cp_compressor_wkv_overlap/REPORT.md')],
    status='scoped_observation')
with path.open('a') as f:f.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
print(entry['id'])
