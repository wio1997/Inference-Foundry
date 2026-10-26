from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
admission = json.loads((ROOT/'evidence/20260928_loop081_bound/run589/final_admission.json').read_text())
assert admission['status'] == 'scoped_same_run_native_partial_rs_copy_identity_admitted'
assert len(admission['capture_rows']) == 8 and len(admission['runtime_hashes']) == 64
assert len(admission['capture_hashes']) == 24
assert all(admission[key] is None for key in ('typed_last_writer',
    'compiled_hc_post_consumer','producer_ready_time','collective_completion_time',
    'consumer_ready_time','scheduling_bound_s','resource_bound_s','product_bound_tps'))

notes = {
'ACHIEVABLE_BOUND.md': '''## Run589 production native slice and independent capacity-method review

Run589's guarded 48+48/c12/1024 fixed-DSpark7 diagnostic and independent final reducer pass. On all8 ranks, one measured cohort5 FULL96 Graph generation joins the same 12 request IDs and runtime cycle0 to source-labeled layer0 BF16 local `wo_b` MatMul (Model45 stream1/task40), a paired-event BF16 ReduceScatter (stream0/task13) and source-labeled attention `MEMCPY_ASYNC` (stream1/task43). Each Graph has 5,412 native tasks; all64 Runtime files, exact24 capture files, 96 client results, tagged stop/all8 idle and five-source restoration are SHA-bound. This replaces Run586's cross-run pointer inference with a scoped same-run **static native identity**. Copy src/dst/count are source-label geometry, not independently exported native parameters; following HC-post remains a candidate, not a typed consumer. Dump `ts/dur` are synthetic, replay ordinal denotes submission, and the post-cohort synchronization does not give node-ready/completion times. No legal shorter schedule or E2E benefit follows.

The parallel Astra High Resource review found the installed 910B device-side HCCS statistics source uses a 500 ms cache refresh timer, u64 software wrap extension and no exported sample timestamp. Its source-to-booted-device implementation identity remains unproved; official HDK26.1 HCCS inventory guidance is not automatically valid for this driver26.0.rc1 build. A 10–20 ms Graph's pre/post HCCS count delta therefore cannot yet be used as physical traffic. Exact-board cumulative compute/HBM/HCCS `C⁺/B`, compulsory fixed-`W₀` work/traffic, full all8 dependency DAG and legal overlap remain open. Formal Current stays Run99 median **571.681 tok/s**; all strict finite Resource/Hardware, Scheduling/Execution and Product E2E endpoints and numerical Current→credible-limit gap remain null. Next priority is rank-local timing and typed consumer on the identified production path with perturbation control, one scoped positive fresh logical W⁻ witness, and HCCS API/device-image/freshness binding before any physical-link traffic experiment. See Run589 evidence, independent reviews and PK-087.
''',
'PERFORMANCE_MAP.md': '''## Run589 same-run static DAG and Resource measurement gate

The all8 measured cohort5 FULL96 Graph now binds source-labeled layer0 BF16 partial MatMul→event-paired RS→source-labeled attention copy to native Model45 tasks in one capture generation. It is a scoped dependency identity, not measured node timing or a proven HC-post typed consumer; copy ABI bytes are not independently exported. The independent Resource review found a possible 500 ms cached HCCS statistics path in installed device-side source, with actual booted-image/API identity still open. This blocks short-window physical-link byte inference until freshness is certified. Next bound-calibrating experiments are a perturbation-controlled production ready/completion timing slice, scoped fixed-W₀ fresh work, and exact-board HCCS measurement semantics. Formal Current571.681 tok/s; strict numerical Bound endpoints and Current→limit distance remain null. See PK-087 and Run589 evidence.
''',
'PROJECT_STATE.md': '''## Run589 Bound checkpoint

Run589 48+48 production diagnostic and independent final validation PASS: same-generation all8 measured cohort5 native layer0 partial→RS→copy identity, exact64 Runtime/exact24 capture, stop/all8 idle and source restore. This closes one static Scheduling identity uncertainty, not producer/collective/consumer time or a Product gain. Astra Resource review separately flags installed 910B HCCS statistics cache semantics as a physical-traffic measurement gate; loaded implementation binding remains open. Next continue rank-local production DAG timing and exact-board capacity/compulsory-work acquisition in parallel. Formal Current571.681 tok/s; strict finite bounds and numerical gap remain unknown.
''',
'RESULTS.md': '''## Run589 same-run native Target TP slice

Guarded fixed-DSpark7 48 warmup+48 measured, 96 responses, all64 Runtime files and exact24 capture files pass final validation. All8 selected FULL96 Model45 Graphs map source-labeled local BF16 partial task1/40→event-paired RS task0/13→source-labeled attention copy task1/43; 5,412 native tasks/rank. Tagged stop, all8 idle and five borrowed-source SHA restores pass. No native ready/completion times, HC-post typed consumer, formal E2E improvement or finite Bound. Formal Current remains571.681 tok/s. See Run589 final admission/findings and independent Resource review.
'''
}
for name, body in notes.items():
    path = ROOT/name
    old = path.read_text()
    assert body.splitlines()[0] not in old, f'duplicate heading in {name}'
    path.write_text(old.rstrip()+'\n\n'+body)

entry = {
 'id':'PK-087',
 'topic':'Production same-run native Target TP slice identity requires measured cohort and Graph generation joins',
 'mechanism':'Source-scoped queue1 native tags after BF16 partial and attention copy, preserved HCCL ExtendInfo, and paired cross-stream events bind one semantic layer0 slice to native task order in a single selected FULL96 Graph generation.',
 'environment':'DeepSeek V4 Flash W4A8, 8×Ascend910B3 DP1TP8, fixed DSpark7; Run589 guarded 48 warmup+48 measured c12/1024 diagnostic, CANN9.1/driver26.0.rc1.',
 'observed':'All8 measured cohort5/cycle0 request joins and Model45 5412-task Graphs admit local BF16 MatMul stream1/task40→event pair→BF16 RS stream0/task13→event pair→source-labeled attention MEMCPY_ASYNC stream1/task43. 96 client, exact64 Runtime, exact24 capture and source restore/idle checks pass.',
 'failure_or_limit':'Native exporter does not independently export copy src/dst/count; labels describe source geometry. HcPost is only a candidate, not a typed consumer. Graph dump timestamps are synthetic and selected replay mark is submission. No per-node timing, compulsory Resource work, shorter legal schedule, formal fixed-W0 trajectory equality or Product E2E gain.',
 'revalidate_when':'For Scheduling Bound, measure identified producer/RS/copy/typed-consumer ready and finish times on the same Graph/replay with matched untagged marker/profiler controls and all8 Host/c12 joins; reject drift in Graph generation, tags, output or work ledger.',
 'extreme_relation':'Removes one production static DAG identity uncertainty while leaving strict finite Resource/Hardware, Scheduling/Execution and Product endpoints and numerical Current-to-limit gap null; Formal Current571.681tok/s.',
 'source':[{'repository':'Inference_Foundry','ref':'run589','path':'evidence/20260928_loop081_bound/run589/final_admission.json'},
           {'repository':'Inference_Foundry','ref':'run589','path':'evidence/20260928_loop081_bound/run589/findings.md'}],
 'status':'scoped_observation'
}
pk=ROOT/'performance_knowledge/entries.jsonl'
ids={json.loads(line)['id'] for line in pk.read_text().splitlines() if line.strip()}
assert entry['id'] not in ids
with pk.open('a') as f:
    f.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
print('Run589 docs and PK appended')
