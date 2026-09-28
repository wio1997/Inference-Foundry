#!/usr/bin/env python3
"""Record the scoped Run600 Product Host accounting without bound promotion."""
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
sections = {
    'ACHIEVABLE_BOUND.md': '''\n## Run600 same-W₀ Product Host envelope\n\nRun600 reuses Run341's 42 hashed inputs and reconciles 32 all8 rank-cohort records with strict execute ordinal, call shape and handoff request-ID gates; Astra's final scoped review passes. In the **same Run341 diagnostic W₀**, cohort first-worker-execute→Runtime-built allrank Host envelopes are 2.879/3.514/3.507/3.214s. Within a rank, the interval outside observed `_model_forward` Host calls ranges 0.594–1.253s across the four cohorts; final 96-token execute-entry→Runtime-built ranges 19.6–121.5ms. These include scheduler work, initial seed/metadata submission, Runtime construction, possible waits and asynchronous device work. They are not idle time, removable time, device-ready completion or a critical-path Bound. Final handoff cached `num_output_tokens` sums 181/175/183/162 include possible async placeholders and are not certified generated/published/client-received prefix counts. Run239, Run597 and formal Run99 have distinct W₀ and remain separate.\n\nAstra's independent Product frontier review makes the next evidence gate explicit: one new fixed W₀ linking all8 residual prefill and initial DSpark seed/KV/state writer readiness to first Target, complete Runtime basis and actual SSE/client delivery, with pre-handoff generated/published/received output ledgers and observer control. No DSpark7 algorithm, acceptance or cycle objective changes. Formal Current571.681 tok/s; finite Resource/Hardware, Scheduling/Execution and Product E2E endpoints and numeric Current→credible-limit gap remain null. See Run600 findings/review and PK-098.\n''',
    'PERFORMANCE_MAP.md': '''\n## Run600 existing-W₀ Product preparation split\n\nA 42-input all8 read-only Run341 reconciliation finds 2.879–3.514s cohort Host first-execute→Runtime-built envelopes and 0.594–1.253s rank-local time outside observed `_model_forward` Host spans. The latter includes nonforward Host operations and async device effects, not an idle or removable Gap. Scheduler handoff counts include possible placeholders and cannot establish delivered p_i. The next acquisition needs same-new-W₀ prefill/initial DSpark KV/device-ready and output-delivery joins; narrow Target-tail overlap remains secondary until the full Product dependency is ranked. Resource compulsory traffic and exact-board mixed C⁺/B stay open. Strict Bound endpoints remain null. See PK-098.\n''',
    'RESULTS.md': '''\n## Run600 same-W₀ Host accounting checkpoint\n\nRead-only Run341-derived reducer: 42 source/evidence inputs, 32 rank-cohort records, all8 request IDs and call ordinal/shape gates PASS; independent Astra scoped PASS. Cohort Host first-execute→Runtime-built envelopes 2.879/3.514/3.507/3.214s, rank-local outside-forward remainder 0.594–1.253s. Scheduler cached output counts can include placeholders. No live service, formal TPS or numeric Bound update. Current Formal571.681 tok/s.\n''',
    'PROJECT_STATE.md': '''\n## Latest Bound checkpoint — Run600\n\nRun600 reconciles the existing Run341 diagnostic Host preparation timeline without mixing W₀ from Run239/597/99. All8/32 records and independent Astra review pass. The observed nonforward interval is unclassified Host/async activity, not proven savings; scheduler output counts may include placeholders. Next priority is one same-new-W₀ Product preparation/device-ready/output-delivery DAG plus observer control, while compulsory Resource work and mixed C⁺/B remain parallel uncertainties. Formal Current571.681 tok/s; strict numerical Bound endpoints null.\n''',
}
for name, section in sections.items():
    path = ROOT / name
    old = path.read_text()
    assert '## Run600 ' not in old and '## Latest Bound checkpoint — Run600' not in old
    path.write_text(old.rstrip() + '\n' + section)

pk = ROOT / 'performance_knowledge/entries.jsonl'
entries = [json.loads(s) for s in pk.read_text().splitlines() if s]
assert entries[-1]['id'] == 'PK-097'
entry = dict(
    id='PK-098',
    topic='Same-W0 Product preparation Host remainder is unclassified, and cached output counts include placeholders',
    mechanism='Join natural Run341 all8 execute marks, ordinal/shape-matched forward Host calls and exact handoff IDs inside one diagnostic W0; subtract rank-local disjoint Host forward union only as accounting.',
    environment='Frozen DeepSeek V4 W4A8, 8×910B3 DP1TP8 DSpark7, warm48→measured48 c12/1024; Run341 original-path diagnostic reused by read-only Run600.',
    observed='42 hashed inputs/32 rank-cohort rows PASS; allrank first-execute→runtime-built Host envelopes 2.879/3.514/3.507/3.214s; rank-local outside-forward 0.594–1.253s; final execute→built 19.6–121.5ms. Astra scoped PASS.',
    failure_or_limit='Host remainder includes scheduler, seed/metadata submission, Runtime build, waits and async device work; no writer-ready or actual SSE ownership join. Cached num_output_tokens may include async placeholders, not generated/published p_i. Instrumented W0 is not formal Run99.',
    revalidate_when='Capture one same-new-W0 client/request→prefill→DSpark seed/KV/state-ready→first Target→Runtime→actual SSE delivery all8 DAG with explicit output ownership and perturbation control; retain fixed algorithm trajectory.',
    extreme_relation='Narrows Current Product Host accounting only; Formal Current571.681 tok/s and finite Resource/Scheduling/Product endpoints and numeric gap remain null.',
    source=[
        dict(repository='Inference_Foundry',ref='run600',path='evidence/20260928_loop081_bound/run600/host_envelope.json'),
        dict(repository='Inference_Foundry',ref='run600',path='evidence/20260928_loop081_bound/run600/astra_review.md'),
        dict(repository='Inference_Foundry',ref='run600',path='evidence/20260928_loop081_bound/astra_product_frontier_next_review.md'),
    ],
    status='scoped_observation',
)
with pk.open('a') as f:
    f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
print('updated four docs and PK-098')
