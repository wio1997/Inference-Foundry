# Run510 — independent terminal and ReduceMean ABI source review

2026-09-27. **PASS for the source findings below; NOT a complete Run502 typed-last-writer or Scheduling certificate.** The general model-to-replay-stream completion contract is supported. The precise Run502 terminal-notify object and ReduceMean output-argument joins remain conditional. No timing, compulsory work, finite Bound, or Current change follows.

Only source retrieval, CPU parsing, file hashing, `readelf` and attempted static disassembly were performed. No service/NPU workload, framework import, patch installation, shared script/model or TaskCtl change. `collect_readonly.py` reproduced eight 5,412-task dumps and checked five selected tasks per rank; it wrote only this evidence directory.

## Provenance and applicability

The installed toolkit reports CANN9.1.0, inner build `V100R001C11SPC001B243`. Fresh file hashes match the earlier pinned ACL/runtime files. `installed_files.json` freezes seven installed files, including the complete matching ReduceMean object, which is archived under `source/`.

| Input | SHA256 |
| --- | --- |
| Installed libascendcl.so | `123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480` |
| Installed libruntime.so dispatcher | `7bc2b610dff552df2f114c95701156b4aba43ffcba24e06e3c249e1cf57bbc21` |
| Installed libruntime_v100.so file | `ab4b573989e6b76e129204cf329bdd916739bdd16092a0efcfa88d8ba011d689` |
| ReduceMean high_precision object | `c549abdc92fe1cbc446fb47872c32112973266b5218948bf88318db91f0f96db` |
| Run469 exact-build NPUGraph.cpp | `dd24abdcce96f81b54f0fc95909c38b9de5925fa8070ac77393c8c5991ebb3d7` |

New primary source is the official [CANN runtime repository](https://gitcode.com/cann/runtime), tag `v9.1.0`, commit **`0b4af8a31c75a1f1f43638a8f75b9e3c6182e031`**. `runtime_source_manifest.json` hashes 22 archived files with original repository paths. These are version-matched public sources, **not an established reproducible-build mapping to installed B243**. The installed binary contains corresponding exporter/capture-model strings; strings and a version number do not establish identical implementation or selected backend. In particular, the v100 file hash is not proof that each historical worker loaded that backend.

`review_input_hashes.json` freezes 32 existing inputs: all eight raw/meta pairs, Run503/506 findings and kernel/source evidence, and the exact-commit torch_npu source plus official docs archived in Run469. Those records preserve the same-acquisition boundaries; this review does not re-admit any invalid run.

## 1. Exporter format: positions can be decoded, roles cannot

At the reviewed torch_npu commit `5dd8ef3f9b375b5ae4a83538d5785754148c3302`, `NPUGraph.cpp:293–298` passes flag1 to `AclmdlRIDebugJsonPrint`. In CANN v9.1.0 `stream.cc:57–70,4859–4890`:

- `CheckAndPrintPlaceHolder` prefixes a label when the current byte offset equals `addrOffset` or `dataOffset`. The label consumes **zero argument bytes**; it annotates the following u64.
- `SetTraceKernelArgs` copies the current device argument buffer to Host and prints `argsSize / 8` u64 words in offset order. It does not provide input/output direction, alias ranges, or a producer identity.
- `DebugJsonPrintForModelStm:4932–4989` assigns `ts`/`dur` from fixed per-task-type constants. They are synthetic visualization values, not device timestamps or measured task durations.

The raw Run502 data independently has 160 bytes / 20 words for each reviewed ReduceMean candidate, on all8 ranks. Placeholder-address labels occur at word2 (byte16) and word6 (byte48). The aux pointer is unannotated word3 (byte24). Placeholder data blobs can occur in different positions: on rank0/task2788 data1 starts at word8 and data0 at16; later candidates reverse their blob order. `raw_argument_checks.json` preserves every word/offset/annotation for the selected tasks. Removing labels preserves offsets; deleting their associated words would corrupt the ABI interpretation.

This establishes the release exporter interpretation and a matching raw-format observation. It does **not** establish that word3 is an output pointer. The copied argument buffer is the post-drain state, not a cycle64 parameter snapshot. Even a decoded output pointer would not exclude later overlapping writes, indirect stores or the 43 MEMCPY_ASYNC tasks whose addresses are absent from this dump.

## 2. ReduceMean MIX: stronger candidate, unresolved lowering

Run503's exact metadata/object still match. The source `reduce_mean.py:178–201` builds the logical tensor list `[x, axes, res]`; metadata names input x/axes and output y. The exact `_2100900_mix_aiv` symbol and its 160-byte `.ascend.meta` section exist in the archived object (`kernel_symbols.txt`, `kernel_metadata_hex.txt`). The installed static disassembler renders the sampled symbol instructions as `<not available>` (`kernel_disassembly.txt`), so no verified machine-code argument/store decoder resulted.

CANN release `program.cc:1080–1119` supplies a concrete reason that logical tensor order may have an additional system argument: under the FFTS_PLUS feature, absent an explicit FFTS metadata flag, MIX/inter-core-sync kernels set `IsNeedSetFftsAddrInArg` and increment `SystemParaNum`; overflow support can also add a system parameter. This count/flag code **does not prove the final insertion order or actual branch for this loaded kernel**. `elf.cc:501–540` parses the legacy DFX argument count; it does not identify semantic x/axes/y directions. The reviewed section does not supply the newer typed parameter-summary records that would by themselves settle the byte24 role.

Thus `word0 = hidden system field; word1 = x; word2 = axes placeholder address; word3 = y` is a plausible ABI hypothesis, not a certified decoder. The leading `0xfee00000000` is not labeled FFTS by this review. Main RmsNorm remains a stronger source-supported writer candidate from Run503; neither it nor the three aux candidates is promoted to an exact final writer.

## 3. Terminal notify to caller: a real release-source mechanism

The official [CANN9.1 asynchronous model API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1822.html) executes the model on the supplied stream. The [same-version cross-stream capture guide](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0031.html) requires child streams to rejoin the capture stream and demonstrates obtaining aggregate outputs by synchronizing the replay stream. This supports the existing Run469 conditional contract; successful Host return alone is submission.

The newly available release implementation explains its mechanism:

1. `model_ri.cpp:25–32` forwards `aclmdlRIExecuteAsyncImpl` to `rtModelExecute(modelRI, stream, 0)`. `CaptureModel::ExecuteCommon:268–310` calls the normal model execution path after setup. `Model::Execute:1258–1290` selects asynchronous stream execution for the ordinary non-forbidden-default stream branch.
2. `model_c.cc:714–758` (`MdlAddEndGraph`) creates/reuses a `Notify`, enqueues `NtyRecord(notify, model_stream)`, and saves the same object as `endGraphNotify`. `CaptureModel::ModelEndGraph:1198–1237` uses the original capture stream for this end marker.
3. `Model::GetStreamToAsyncExecute:1181–1236` sets the execution stream, submits model execution, then enqueues **`NtyWait(endGraphNotify_, execution_stream, ...)`** for Stars/non-AICPU execution, including both software-SQ and ordinary capture variants. Consequently subsequent work on that execution stream is ordered after the model end notification under those branches.
4. `SetNotifyAfterExecute` is a separate mechanism: it records on the execution stream and waits on added streams. Its name must not be substituted for the endGraphNotify wait above.

The exporter `model_aclgraph.cc:50–80` traverses the model's bound `streams_`; the caller execution stream may differ from capture/model streams. Therefore absence of the caller wait from the graph dump is compatible with this implementation. Run503's reconstructed graph still has no reverse path from s1/notify to the already-ended internal s0. **No equal-number mapping from internal s0 to caller logical stream0 is needed or justified.**

**Admit:** a source-supported general producer-in-model → model completion → subsequent replay-stream event contract, with an explicit release implementation. **Keep conditional for Run502:** whether `NOTIFY_RECORD_2240` is exactly that model's endGraphNotify, the actual loaded backend/build and branch, and same-generation producer membership plus actual caller R1 submission order. Do not demand a broad hardware experiment merely to rediscover the documented general contract. Do not claim the precise historical native edge was observed.

## Minimum next certificate and Bound gate

1. **Finish the source join first.** Obtain an authoritative B243 source/build mapping for the selected runtime backend, or a supported native correlation exposing model/endGraphNotify/execution-stream identity. For MIX, obtain the exact compiled launch-layout/argument descriptor including system fields, placeholder relocation and workspace/tiling insertion; bind it to this object/symbol. A guessed word shift is insufficient. These are two bounded missing records, not a reason to rerun broad timing.
2. **Bind semantic outputs during an ordinary capture if existing evidence cannot supply them.** At the three actual aux mean returns and final norm return, retain scalar tensor/storage descriptors, model/layer/role, selected entry and generation; pin the loaded model source. Use capture correlation to bind these returns to native tasks and loaded object/ABI. Include all overlapping stores/copies through terminal completion before calling any task the last writer. Do not retain extra strong tensor references that change graph-pool lifetimes.
3. **Only if exact native terminal evidence remains necessary**, narrowly observe the existing model execute, its end notify record/wait, actual replay stream and subsequent R1 EventRecord with generation/ordinal correlation. Preserve ordinary calls; no extra waits or selected-path native stream getters. A source-proven general contract plus bound membership/order can support the conditional theorem; a targeted trace supplies missing actual-run identities, not a time lower bound.

Run502's private MLA update execution count/event generation remains unproved by this review. A separately admitted later run may close its own zero-zip case; it cannot be backfilled into Run502. Positive external updates still require the same-generation event consumer relation. Raw address occurrence, exported synthetic durations and post-drain state cannot substitute for any of these joins.

Even after completion ownership closes, Scheduling needs the mandatory dependency scope, all-rank arrival alignment, allowable concurrency/resource service, and a justified necessary/attainable schedule. Source order or measured replay envelopes alone do not prove a compulsory cycle duration. Resource W-minus/C-plus gates remain separate. **Proceed with the bounded source/identity certificate work; do not promote Run502 candidates or produce a numerical ceiling.**
