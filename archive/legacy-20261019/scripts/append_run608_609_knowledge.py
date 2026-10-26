#!/usr/bin/env python3
"""Append scoped dual-Bound and two-cycle packet knowledge, once."""
import json
from pathlib import Path


ROOT = Path('/data/wio/Inference_Foundry')
DEST = ROOT / 'performance_knowledge/entries.jsonl'
ENTRIES = [
    {
        'id': 'PK-103',
        'topic': 'Separate conditional attained service from strict fixed-work Resource and Scheduling certificates',
        'mechanism': 'A positive strict resource time floor needs a matched necessary fixed-W0 amount W_minus and an exact-board cumulative no-faster-than service envelope C_plus with finite burst B. A Scheduling floor additionally needs actual all8 data/stream/storage generations and legal overlap. Observed feasible service belongs only to conditional Engineering calibration.',
        'environment': 'DeepSeek V4 Flash W4A8, 8×910B3 DP1TP8, frozen DSpark7 acceptance/cycle/output; Run607 independent dual-Bound audit and V3.37 Run608.',
        'observed': 'Run606 closes ordinary current issued geometry and Graph dispatch for one diagnostic W0. Run607 finds no matched W_minus/C_plus,B certificate or complete all8 legal execution DAG. V3.37 independently rebuilds all pinned raw and keeps Resource/Scheduling/Product strict endpoints and numeric Current-to-credible-limit gap null; formal Current571.681 tok/s.',
        'failure_or_limit': 'Attained kernel/HCCL rates are lower evidence of feasible service, not upper hardware C_plus. Current 18.965GB read/2.380GB write, context rows or 49,152 output IDs are not compulsory bytes/work. Diagnostic profiler timing and historical scenarios cannot define a ceiling.',
        'revalidate_when': 'Prove one declared fresh semantic subset and exact-board clock/issue/HBM/HCCS C_plus,B with source-to-loaded-image and counter freshness; build one full all8 producer-consumer cycle DAG and matched mixed-service observer control. Maintain conditional Engineering assumptions separately.',
        'extreme_relation': 'Maintains Bound-first priority without targeting speculative acceptance or cycle reduction; avoids manufacturing a numerical distance from571.681 before proof.',
        'source': [
            {'repository': 'Inference_Foundry', 'ref': 'run607', 'path': 'evidence/20260928_loop081_bound/run607/astra_bound_review.md'},
            {'repository': 'Inference_Foundry', 'ref': 'run608', 'path': 'evidence/20260928_loop081_bound/run608/bound_calibration_v3_37.json'},
            {'repository': 'Inference_Foundry', 'ref': 'run608', 'path': 'evidence/20260928_loop081_bound/run608/astra_model_review.md'},
        ],
        'status': 'scoped_observation',
    },
    {
        'id': 'PK-104',
        'topic': 'Same-W0 Draft context/query numeric slot alias is a layout hazard hypothesis, not a mandatory serial edge',
        'mechanism': 'Run609 pins Run606 all8 cohort5 cycle64/65 prior W0 and compares actual sparse Draft context/query slot labels per request. The intersection for cycle64 equals 8 minus its accepted count for each request in groups2/3. A legal storage redesign could remove alias even if current stores/readers share cache.',
        'environment': 'Run606 observer-perturbed fixed DSpark7 diagnostic W0, cohort5 cycles64/65, all8 ranks, current eager Draft context96/query84, Q7. New W0 acceptance/slot counts may differ.',
        'observed': 'Cycle64 accepted47 and cycle65 accepted59 in the prior W0. Each current group2/3 has49 overlapping numeric context/query slot labels at cycle64; per-request intersections follow 8-minus-accepted. Astra independently rebuilt source/raw gate and passed scoped design review.',
        'failure_or_limit': 'Sparse cycle64 labels do not prove per-layer scatter, cache storage identity, actual first query read, stream readiness or necessary cross-cycle edge; cycle65 slot witness is absent. Prior branch_history does not prove actual FULL Graph replay. There is no timing result or finite Scheduling Bound.',
        'revalidate_when': 'New all8 W0 cohort5 cycles64/65 with actual Graph/native generation; bind KV writer/storage versions and query first consumer, Host count-copy63-to64/64-to65, next Target65 and Target66 boundary. Keep new W0 acceptance descriptive, no transfer of47/59 or49 as a gate.',
        'extreme_relation': 'Prioritizes a same-state storage-generation and scheduling measurement while retaining freedom for space-time layout changes; no formal E2E improvement claim.',
        'source': [
            {'repository': 'Inference_Foundry', 'ref': 'run609', 'path': 'evidence/20260928_loop081_bound/run609/two_cycle_packet_gate.json'},
            {'repository': 'Inference_Foundry', 'ref': 'run609', 'path': 'evidence/20260928_loop081_bound/run609/astra_preflight_review.md'},
            {'repository': 'Inference_Foundry', 'ref': 'run606', 'path': 'evidence/20260928_loop081_bound/run606/live/b/basis_admission.json'},
        ],
        'status': 'scoped_observation',
    },
]


def main():
    existing = [json.loads(line) for line in DEST.read_text().splitlines() if line.strip()]
    ids = {row['id'] for row in existing}
    if ids & {entry['id'] for entry in ENTRIES}:
        raise ValueError('knowledge ID already exists')
    with DEST.open('a') as handle:
        for entry in ENTRIES:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
    print('added', ','.join(row['id'] for row in ENTRIES))


if __name__ == '__main__':
    main()
