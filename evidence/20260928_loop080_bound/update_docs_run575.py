#!/usr/bin/env python3
"""Append the source-scoped Run575 Graph kernel join once."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
APPENDS = {
    'ACHIEVABLE_BOUND.md': '''
## Run575 selected FULL96 Graph native branch association

Run502's admitted selected FULL96 Graph has exactly43 native `SparseAttnSharedkv` tasks per rank. All8 ranks select the Run574 packaged BF16 object prefix `7d53…`; installed CANN9.1 key encoding maps per-rank keys2/514/1026 (2/20/21 tasks) to `FLASH_DECODE=0`, Query TND, KV PA_ND, SWA/CFA/SCFA. This closes the prior ambiguity about the *exported selected Graph's kernel name and static tiling branch*, not the loaded device bytes or dynamic query-prefix/head state. Complete DSA/MoE row mapping, same-trajectory W₀ work, cumulative capacity C⁺/B and legal all8 Scheduling DAG remain open; finite Resource, Scheduling and Product endpoints and numeric Current→limit distance remain null. Current Formal571.681tok/s. See Run575 findings/Astra review.
''',
    'PERFORMANCE_MAP.md': '''
## Run575 Graph/package branch evidence

An eight-rank Run502 Graph→Run574 package join identifies 344 selected BF16 sparse-attention tasks, 43/rank, with key distribution SWA2/CFA20/SCFA21 and static FD0/TND/PA_ND branch under the installed CANN9.1 encoder. This supports the native Query-row preservation hypothesis for that Graph, but dynamic prefix/head/group, actual loaded object bytes and all-layer router provenance still need an admitted same-trajectory witness. It is structural Bound calibration, not a latency or savings measurement. See PK-076.
''',
    'PROJECT_STATE.md': '''
## Run575 selected Graph native branch checkpoint

Offline Run575 and Astra independent replay PASS. The selected Run502 FULL96 Graph's all8 native sparse attention task names join the packaged BF16 object and CANN9.1-decoded FD0/TND/PA_ND templates (2 SWA,20 CFA,21 SCFA per rank). The dump is structural after later replays, not cycle64 dynamic metadata; no service/NPU run occurred. Formal Current571.681tok/s, every finite Bound endpoint and the numerical Current→credible-limit distance remain null. Next Resource witness: actual first-post-parking branch/prefix/group, all43 semantic router rows and Target/Draft packed formats in one all8 trajectory; independently pursue fixed-W₀ mixed service and exact-board capacity.
''',
    'RESULTS.md': '''
## Run575 offline Graph/package/tiling join (2026-09-28)

All8 admitted Run502 FULL96 Graph dumps contain 43 sparse-attention MIX_AIC tasks/rank. Their names join one installed BF16 package object; keys2/514/1026 decode to FD0/TND/PA_ND and SWA/CFA/SCFA, with counts2/20/21 per rank. Astra independently reproduced the output and negative gates. No model run, performance measurement or numeric Bound promotion occurred. Formal Current remains571.681tok/s. Evidence: `evidence/20260928_loop080_bound/run575/`.
''',
    'HANDOFF.md': '''
## Run575 Bound checkpoint

Run575 source-pinned offline Graph/package/CANN-key join and Astra review PASS. The selected FULL96 Graph structurally chooses 43 BF16 sparse-attention tasks/rank with FD0/TND/PA_ND SWA2/CFA20/SCFA21 branches; actual device-loaded bytes, dynamic prefix/head metadata and 43-layer semantic row map remain open. No NPU service was started. The next live Resource acquisition should be one guarded all8 first-post-parking same-trajectory row/expert/weight witness; Resource capacity and Scheduling mixed-service/critical-path tracks continue independently. Strict numerical Bound endpoints and Current→limit distance remain null; Formal Current571.681tok/s.
''',
}

for name, section in APPENDS.items():
    path = ROOT / name
    old = path.read_text()
    marker = section.splitlines()[1]
    if marker in old:
        raise ValueError(f'already appended: {name}')
    path.write_text(old.rstrip() + '\n\n' + section.strip() + '\n')

path = ROOT / 'performance_knowledge/entries.jsonl'
old = path.read_text()
if '"id": "PK-076"' in old:
    raise ValueError('PK-076 already exists')
entry = dict(
    id='PK-076', topic='Selected FULL96 Target Graph names identify BF16 native sparse-attention templates',
    status='conditional_graph_structure',
    environment='DeepSeek V4 Flash W4A8 8x910B3 DP1TP8 DSpark7; admitted Run502 selected FULL96 Graph after later replays; Run574 installed package; CANN9.1 key encoder.',
    mechanism='Graph exporter task names contain packaged kernel basename and tiling key. Installed CANN encoder appends BOOL1/UINT4 declaration indices, yielding FD0/TND/PA_ND SWA/CFA/SCFA for selected keys.',
    observed='Run575 source-pinned all8 join: 43 tasks/rank, 344 total; package object 7d53… BF16 q/output; keys2/514/1026 counts2/20/21 per rank. Independent Astra byte-identical replay and mutation gates pass.',
    failure_or_limit='Name/key join does not prove actual device-loaded object bytes, source→object reproducible build, dynamic query-prefix/head arguments, layer ordinal, 43-layer row identity, compulsory traffic, latency or Product Bound. Run502 dump is post-replay structural data, not cycle64 parameters.',
    revalidate_when='Graph capture generation, kernel package, CANN encoder or DeepSeek config changes; bind actual selected runtime values and all8 same-trajectory first-post-parking row/weight witness before semantic expert-incidence use.',
    extreme_relation='Narrows one native branch of Resource row-map uncertainty; Current Formal571.681tok/s and all finite Resource/Hardware, Scheduling/Execution and Product endpoints unchanged.',
    source=[dict(repository='Inference_Foundry', path='evidence/20260928_loop080_bound/run575/graph_kernel_join.json', ref='all8 selected Graph/package/tiling join'),
            dict(repository='Inference_Foundry', path='evidence/20260928_loop080_bound/run575/astra_review.md', ref='independent review'),
            dict(repository='Inference_Foundry', path='evidence/20260927_loop079_identity/run502/b_candidate/graph_dump_admission.json', ref='admitted selected production Graph')],
)
with path.open('a') as f:
    f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + '\n')
print('Run575 docs and PK-076 appended')
