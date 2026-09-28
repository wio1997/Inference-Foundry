from __future__ import annotations

import json
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
notes={
'ACHIEVABLE_BOUND.md':'''## Run586–588 native identity methodology checkpoint

Run586 independently reduces Run502's all8 FULL96 native Model45 Graphs to a static MatMul→event-record/wait→BF16 RS→event-record/wait→copy→HC-post candidate chain; eight SHA-bound 5,412-task dumps and six negative mutations pass. The dump's `ts/dur` are synthetic. Run583's partial/reduced/destination addresses each recur 82/41/372 times in every Run502 dump, so cross-run pointer matching cannot certify last writers. Run587's direct Python `aclrtCacheLastTaskExtendInfo` failed on queue1 because its caller-thread last-stream TLS was unset (`65535`); this was an instrumentation failure, not a Graph prohibition. Run588's exact-header C++ `OpCommand::RunOpApi` sidecar passed an isolated queue1 Graph: three labels attached exactly to MatMul, `MEMCPY_ASYNC` copy and Add on Model49/stream13, with correct output and all8 idle after exit. Astra independently verified source/binary/dump identity.

The production transfer remains unproved. A Run589 guarded acquisition now tests whether source-scoped partial/copy labels, request/cohort/cycle identity and same-generation Graph dump can join native RS event order without overwriting HCCL metadata or entering compiled decoder forward. Until final admission, producer/completion times, HC-post typed consumer, legal schedule, compulsory Resource work, strict exact-board `C⁺/B` and every finite Resource/Scheduling/Product endpoint remain unknown. Formal Current stays Run99 median571.681 tok/s. See Run586–588 evidence and PK-085/086.
''',
'PERFORMANCE_MAP.md':'''## Run586–588 native DAG mapping method

Run586 recovers a historical all8 static MatMul→RS→copy→HC-post event-order candidate from Run502, but synthetic dump times and reused Graph-pool addresses cannot bind Run583 or quantify a scheduling window. Run587 rejects a direct Python TLS tagging route on current queue1. Run588 establishes a source-compatible queue1 C++ sidecar tagging method on an isolated Graph, including native `MEMCPY_ASYNC`. Production applicability is the Run589 question; no Current→Bound time shrinks yet. Formal Current571.681 tok/s; strict finite endpoints stay null. See PK-085/086.
''',
'PROJECT_STATE.md':'''## Run586–588 native identity checkpoint

Run586 offline all8 Run502 static event chain PASS, no typed cross-run lineage or time. Run587 Python native-tag route INVALID (`lastStreamId=65535`); all8 idle after. Run588 isolated queue1 C++ sidecar PASS: exact MatMul/copy/Add native task labels, output and clean exit; Astra independently checked ABI and SHA. Run589 guarded production identity acquisition is the next Scheduling Bound discriminator. One-row conditional Resource W⁻ Run585 preflight remains parallel; exact-board `C⁺/B` is missing. Formal Current571.681 tok/s; numerical credible-limit gap remains unknown.
''',
'RESULTS.md':'''## Run586–588 native identity evidence

Run586 offline static all8 Model45 event-order chain PASS; no native time. Run587 direct Python Graph tag INVALID due to unset queue1 caller-thread last-stream TLS, no service/client. Run588 one-card queue1 C++ sidecar PASS: three unique labels on MatMul, `MEMCPY_ASYNC`, Add (Model49/stream13/tasks0–2), correct output, all8 idle after; not a production result. Strict finite Resource/Scheduling/Product endpoints remain unknown; Formal Current571.681 tok/s.
'''
}
for name,body in notes.items():
    path=ROOT/name;s=path.read_text();heading=body.splitlines()[0]
    if heading in s:raise RuntimeError(f'duplicate heading {name}')
    path.write_text(s.rstrip()+'\n\n'+body)

pk=ROOT/'performance_knowledge/entries.jsonl'
ids={json.loads(s)['id'] for s in pk.read_text().splitlines() if s.strip()}
entries=[
{'id':'PK-085','topic':'Historical FULL96 Graph event order is useful, but cross-run addresses do not bind native writers',
 'mechanism':'Exact Model/Stream/Task and paired events recover a static native order; Graph-pool address reuse and synthetic dump timestamps prevent storage lifetime or time promotion.',
 'environment':'DeepSeek V4 Flash W4A8, 8×Ascend910B3, DP1TP8, fixed DSpark7; Run586 offline Run502 selected FULL96 cohort5 Graphs and separate Run583 capture.',
 'observed':'Eight SHA-bound 5412-task Model45 dumps show stream1 MatMul task40→event41→stream0 wait12→BF16 RS13→event15→stream1 wait42→MEMCPY_ASYNC43→HcPost44. Six negatives rejected. Run583 partial/reduced/destination pointer texts recur 82/41/372 times per Run502 rank dump.',
 'failure_or_limit':'Run502 graph does not independently bind layer0 semantic operations or Run583 storage. MEMCPY_ASYNC lacks exporter src/dst/count; dump ts/dur are synthetic. Run378 ordinal3 and Run580 timing belong to other runs. No actual readiness or finite Scheduling/Product Bound.',
 'revalidate_when':'Capture source-labeled producer and copy native tasks, same-generation graph ownership, measured request/cycle and HCCL event-chain in one production run; measure rank-local completion only after identity and marker perturbation gates.',
 'extreme_relation':'Narrows static dependency topology and rules out cross-run pointer-join shortcut; strict finite Bound endpoints and numeric Current-to-limit gap remain null, Formal Current571.681tok/s.',
 'source':[{'repository':'Inference_Foundry','ref':'run586','path':'evidence/20260928_loop081_bound/run586/static_event_dag.json'},{'repository':'Inference_Foundry','ref':'run586','path':'evidence/20260928_loop081_bound/run586/findings.md'},{'repository':'Inference_Foundry','ref':'run583','path':'evidence/20260928_loop081_bound/run583/final_admission.json'}],
 'status':'conditional_graph_structure'},
{'id':'PK-086','topic':'Installed queue1 native Graph task labels require submission-thread C++ enqueue, not Python last-task TLS',
 'mechanism':'aclrtCacheLastTaskExtendInfo uses calling-thread last stream/task TLS. A Python ctypes caller may be off the queue1 submission thread; torch_npu OpCommand RunOpApi enqueues a value-owned callback on that thread and can label the immediately preceding Graph task.',
 'environment':'torch_npu2.10.0.post4/CANN9.1 on Ascend910B3; Run587 direct Python invalid probe and Run588 one-card BF16 queue1 Graph sidecar preflight; no model service.',
 'observed':'Run587 rc507000 with CANN stream_id65535/null stream. Run588 exact installed-header C++ sidecar labels Graph Model49/stream13 task0 MatMul, task1 MEMCPY_ASYNC and task2 Add uniquely in order; output correct, exit0, all8 idle. Astra independently verified source/dump hashes and libtorch_npu/libascendcl linkage.',
 'failure_or_limit':'Isolated queue1 PASS does not prove multi-stream production last-task binding, unperturbed Graph capture, fixed-W0 trajectory, actual copy ABI or timing. Tagging an HCCL task could overwrite its original ExtendInfo; compiled decoder Python hook previously broke fullgraph.',
 'revalidate_when':'Run one guarded production FULL96 same-generation capture with source-role tags only after local matmul and attention copy, preserve HCCL group metadata, join unique measured cohort/request/cycle and inspect exact native tag landing. Reject unexpected task type/order or output drift before any timing claim.',
 'extreme_relation':'Provides a reusable native identity measurement method for Scheduling Bound acquisition; does not change Current571.681tok/s or any finite strict endpoint.',
 'source':[{'repository':'Inference_Foundry','ref':'run587','path':'evidence/20260928_loop081_bound/run587/cann_error_excerpt.log'},{'repository':'Inference_Foundry','ref':'run588','path':'evidence/20260928_loop081_bound/run588/result.json'},{'repository':'Inference_Foundry','ref':'run588','path':'evidence/20260928_loop081_bound/run588/findings.md'}],
 'status':'scoped_observation'}]
for entry in entries:
    if entry['id'] in ids:raise RuntimeError(f'duplicate PK {entry["id"]}')
with pk.open('a') as f:
    for entry in entries:f.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
print('docs/PK Run586–588 appended')
