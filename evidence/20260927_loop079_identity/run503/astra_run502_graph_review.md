# Run503 — independent Run502 acquisition and graph-structure review

2026-09-27. **ACCEPT Run502 as a valid, source-bound, post-drain graph diagnostic and conditional Current structure evidence.** **Do not yet admit four proven last writers, effective private-stream no-work/completion, or an unconditional producer→caller-R1 certificate.** No finite Bound, removable time, formal Current change or optimization conclusion follows.

This review did not launch/query a service, use an NPU workload/probe, install a patch, modify shared sources/scripts, change TaskCtl or update a model. CPU analysis and read-only installed-container file retrieval were used; only Run503 evidence was written.

## Acquisition and integrity certificate

- Independently reran `validate_root` and `final_admit`: both pass. UUID is `535915a1-4c13-418d-aeec-988d2e6e000b`. The complete40 capture +40 Runtime rows, eight raw graph dumps and eight graph meta files form the same acquisition. Native raw SHA/bytes/task/stream metadata, selected entry/generation/graph/model/batch, output-owner digest and bound debug method receiver all join.
- `input_hashes.json` pins **145 inputs**: all135 candidate files plus the actual server log, backend source and eight reviewed scripts. Every hash was rechecked at review end. There was no preexisting universal145-file manifest to “match”; raw hashes match their acquisition metadata, reviewed scripts/source pins match their prior certificates, and the new full manifest freezes all other audit inputs.
- Within each rank, entry ID, graph ID, capture generation1, descriptor and output tree remain stable across cohorts1–5; selected observation ordinals are1–5. The graph is FULL96/12-request, main Target, startup capture bound to the same Runtime model and argument/output storage. Same numeric IDs on different ranks/runs are not compared as object identity.
- Actual server log has exactly60 chat POSTs, all200. Both client files independently give60 total requests with no recorded error and exactly1024 output tokens each. The combined request sweep peaks at12; warmup completes before the diagnostic. Cohort cycles are308/321/294/325/303. All-rank Runtime counts/request/trajectory and ownership checks pass. These are correlated diagnostic cohorts, not formal timing repeats or a new semantic token-text oracle.
- All eight cleanup status fields are0. Six live restored source hashes and all six archived backups match original pins; before/after SHA files agree. There are eight ERROR-prefixed server lines at11:13:15 from `AsyncLLM output_handler`/`EngineDeadError`, after “Parent process exited” and shutdown cleanup messages. They are preserved in `shutdown_log_excerpt.txt`. They occur in the successful controller's stop context, after completed requests/validation; this review does not call the server log error-free or silently discard them.

## Actual graph-update backend: bound, but effective work still unknown

All eight metadata records bind the successful selected `after` call to `vllm_ascend.attention.mla_v1.AscendMLAImpl.update_graph_params`, with the setup backend/getter, actually resolved implementation/update callable and post-drain source agreeing. Installed `mla_v1.py` hashes to `67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545`, exactly the recorded source hash; an unchanged copy is archived here. The selected call returned once. Configured update stream is logical102 on each own rank.

This is **not** the no-op `AscendDSAImpl.update_graph_params` in `dsa_v1.py`. The actual MLA source at968–1097 can return for zero attention keys; with nonempty keys it iterates the zip of keys/parameters/handles/events. The zip can also be empty. When iterations execute, source enters `torch.npu.stream(update_stream)` and calls graph_task_update_begin, FIA.out, graph_task_update_end and event.record. Run502 records neither attention-key/zip lengths nor actual iteration/API counts. Therefore neither “stream102 did no work” nor “each layer executed private-stream update work” is certified.

The corresponding MLA capture branch at1884–1945 creates an ExternalEvent, places wait/reset on the capture stream, appends that event and graph-task-group handle, and captures FIA. **If** the selected update uses those same event/handle objects and the relevant generation, an after-submission update/record can release a wait in the already submitted same replay. Host “after replay() returned” does not mean “after graph execution completed,” and this cannot safely be labeled only next-round preparation. Conversely, source alone does not establish that these particular objects/iterations were used in Run502.

The exported graph does not contain stream102, and all exported waits have exported record-name matches. That does not establish zero external update work or identify those matches with the Python ExternalEvents. No event-object/native-event/handle/generation mapping for the actual update was recorded. Keep effective update work and its consumer relation unresolved.

## Native graph structure

All eight graphs have5,412 tasks, native model45, and stream0/1/99 counts1,040/3,985/387. Exact task identities are unique, contiguous per stream, and pass the installed-format schema. Each contains1,038 EVENT_RECORD,735 EVENT_WAIT,1,038 EVENT_RESET,43 MEMCPY_ASYNC and one NOTIFY_RECORD, plus2,557 kernel tasks. MEMCPY_ASYNC exposes no source/destination/byte fields.

Event suffix IDs are unique within each type; every wait has one record of the same suffix, and record/reset sets coincide. There are303 records with no dump-local wait. Candidate record→wait stream pairs are1→99:172,99→1:43,0→1:260 and1→0:260. Those observations do not prove event reuse/reset-generation semantics or that a record with no wait is redundant.

Taking ascending per-stream Task Id as sequence and matching record/wait suffixes as cross-stream edges yields an acyclic candidate DAG. All5,412 tasks can reach the terminal notify at stream1 task3984; notify suffix differs by rank. This is a structural reconstruction under those explicit ID/order assumptions, not a measured schedule, common clock or latency. No exporter `ts`/`dur` was used as time evidence.

**The internal graph-to-caller join remains separate.** Graph stream0 ends at task1039 (`EVENT_RECORD_5252` on rank0); stream1 continues to main output candidate2944 and terminal notify3984. Main2944 and notify3984 have no path back to the already-ended graph stream0. The direction matters: graph stream0 can precede later stream1 work; lack of a reverse edge is not a deadlock finding. The caller marker uses logical stream0, but same-number identification with the dump's internal stream0 cannot establish R1 completion. The model is submitted by source-pinned NPUGraph::replay through `aclmdlRIExecuteAsync(model_ri_, current_stream)`; the dump does not expose its terminal model-notify→caller continuation link. R1 covers a required producer only after that producer's membership and the relevant model/external-event completion contract are established.

## Four output addresses: candidates, not last-writer proof

All four captured output leaves are BF16[12,4096], contiguous, zero offset,98,304-byte storage. Exact address matches occur repeatedly in kernel text because graph memory is reused. The same four unique maximal **address occurrences** appear on all ranks under the candidate DAG:

| Leaf | Rank0 data pointer | Exact-address occurrences | Maximal occurrence | Raw hexadecimal word position, zero-based |
| --- | --- | ---: | --- | ---: |
| main | 0x130800086800 |130| stream1/task2944, RmsNorm |2|
| aux0 | 0x1308002e6e00 |372| stream1/task2788, ReduceMean |3|
| aux1 | 0x130800175000 |278| stream1/task2848, ReduceMean |3|
| aux2 | 0x13080018d200 |321| stream1/task2925, ReduceMean |3|

Searching all literal addresses within each full output storage range produces the same maximal candidates. All four reach terminal notify in the reconstructed graph. This search is stronger than selecting the last JSON row, but **a maximal occurrence is not a typed write**. It misses indirect addressing, pointer tables, runtime-patched arguments and MEMCPY_ASYNC addresses that the exporter does not expose. Neither the number of matches nor address equality determines input/output role or excludes a later hidden writer.

Read-only installed metadata strengthens the candidates:

- The exact kernel name `RmsNorm_normal_all_bf16_high_performance_0` appears in installed RmsNorm metadata. Its object SHA `e786e2864dc63bcbec96231eb062eb81646859e96bd759e59b3dad6a5bf759c2` matches the metadata's binary SHA. Metadata lists x/gamma inputs and y/rstd outputs; archived `rms_norm.cpp:46–47` declares the third GM_ADDR as y. Main pointer occupies the third unannotated leading address in this dump. This is strong matching source/ABI evidence for a writer candidate.
- The exact ReduceMean name including `_high_precision_2100900` appears in the installed legacy binary metadata; object SHA `c549abdc92fe1cbc446fb47872c32112973266b5218948bf88318db91f0f96db` matches its metadata. Metadata declares x/axes inputs and y output. The dump is `KERNEL_MIX_AIV` with an extra leading address and placeholder annotations; raw word position3 is not automatically the third API argument. A typed decoder for that actual compiled/placeholder ABI has not been proved.
- Current `deepseek_v4.py:1139–1140` appends `hidden_states.mean(dim=1)` to aux outputs;1172–1174 returns final norm hidden plus aux list. Its archived current source SHA is `11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247`. This explains the semantic candidate operators but was not independently pinned as the selected model's loaded source in Run502, so it is corroboration, not a replacement for same-capture typed provenance.

Installed metadata/source/binary correspondence does not itself bind a runtime-loaded binary address to this acquisition or prove no subsequent write. Main is a stronger ABI-supported candidate; all four remain short of the requested exact last-writer certificate.

## Minimal next certificate

First finish a **source-only typed decoding** of the exact installed ReduceMean mixed-core/placeholder argument layout and model-execution terminal-notify semantics. Reuse the archived matching metadata and object hashes; do not infer positions by deleting placeholder strings without an exporter/ABI contract.

If the remaining joins require dynamic evidence, use one new all8 acquisition with the existing frozen shape, cycle64 selector, normal counts drain and cleanup gates. Add only the following targeted records; preserve ordinary calls and add no diagnostic waits/raw stream getters:

1. **Actual update execution ledger.** At the original selected `update_graph_params`, record the actual draft/main flags, `num_tokens`, attention-key count, lengths of the exact params/handles/events lists, the actual zip iteration count, and successful begin/FIA/end/event.record counts. Preserve the original zero-key return and distinguish zero-key/zero-zip/no-op from positive work. Record Host submission order only; no latency claim.
2. **Same-generation event/handle ledger.** During original capture, retain strong references/identities for each ExternalEvent and task-group handle, associated layer, output descriptors and final entry/capture generation. At each selected update iteration record those same identities, replay ordinal and immutable configured stream102. To join Python objects to exported native event/stream/task identities, use a supported native correlation record at the existing wait/reset/group/update/record calls, or a separately labeled targeted native trace that observes their existing handles. Do not compare Python `id(event)` to an EVENT_WAIT suffix. Track reset and replay generations explicitly. Then test whether update record releases this replay's wait, the next replay, or no captured wait.
3. **Typed output producer ledger.** On the actual ordinary FULL capture, record the return tensor descriptors directly at the three existing aux `mean(dim=1)` calls and final norm return, including model/layer, output role, storage/range, graph entry and generation. This identifies semantic producers without guessing from raw addresses. Bind those operations to actual native tasks and loaded kernel/ABI through capture correlation; record all overlapping output writes, including memory copies/in-place aliases, until graph completion. A targeted native trace is needed only for unresolved task/argument/terminal joins. A same-generation producer→terminal→caller-R1 chain must be demonstrated before declaring completion.

If update iteration count is zero, that narrowly closes no effective update work for that selected call; no stream-timing experiment is needed for it. If positive, preserve it as required-until-proved-otherwise and correlate its ExternalEvent consumer in the in-flight graph. A caller R1 marker downstream of model completion may cover the dependency once those identities/generations and runtime semantics are proved; Run502 alone cannot establish that.

## Admission boundary

Admit now: a valid all8 acquisition, same-process selected graph/output ownership, actual selected backend/callable source, supported task vocabulary, exported event structure and ABI-supported producer candidates. Keep conditional: typed last writers, external update/event generation, model terminal→caller continuation and post-drain→cycle64 parameter correspondence. Keep unknown: required work/traffic, resource capacity, unattainable/removable time and any numerical ceiling. Formal Current and all finite Bound endpoints stay unchanged.

Artifacts: `independent_results.json`, `input_hashes.json`, `astra_audit.py`, `shutdown_log_excerpt.txt`, archived `mla_v1.py`, `deepseek_v4.py`, `rms_kernel.json`, `mean_kernel.json`, `rms_norm.cpp` and `reduce_mean.py`.
