# Independent H9 stream-order review

Source-only Challenger review, 2026-10-07. No service, device request, deployment, or frozen Run changed. H6/H5 remain the active verified research stack. This review is not KEEP or a measured gain.

## Verdict and evidence

**KEEP the narrow question for CPU contract validation; no source-proven blocker found for the reviewed resident target path. Global priority and benefit remain unknown.** The diff removes only the Ascend breakable wrapper's pre-replay host wait. It does not remove device dependencies, base offloader fencing, or the capture path.

Personally read the two-file diff, original ModelRunner/ForwardContext/Ascend context/base and Ascend wrappers/NPUGraph.cpp, the inherited GPU runner input-preparation fence, and SFA's no-op graph-parameter update. In ModelRunner, sparse SFA skips `_update_full_graph_params_if_needed`; no auxiliary host task-parameter update is being made by this function. Input and attention metadata construction are inside `synchronize_input_prep`; the inherited async path waits for its prior event before reusing CPU staging and records that event after preparation. The patch retains this protocol. Synchronous scheduling returns through synchronized output bookkeeping. Rotary updates and MC2 mask writes remain before forward on the current stream. These facts support device ordering without this additional host wait; they do not establish safety for arbitrary external concurrent mutation.

The base `_replay` still calls `get_offloader().sync_prev_onload()`. Native `NPUGraph::replay` uses `AclmdlRIExecuteAsync` on the current stream, and retains its own blocking-mode conditional synchronization and RNG replay prologue. Existing captured HCCL stream/event dependencies are untouched. No evidence makes a host wait necessary merely because HCCL uses an internal stream. Conversely, clean Run265 with the old wait does not itself verify its removal.

The mutable context property is assigned a fresh Boolean on every `_model_forward`, including false for revoked conditions. The wrapper also requires FULL and non-draft. Missing properties default false. A cold capture never enters `_replay`, so capture setup is unchanged. The current diff only affects BreakableACLGraphWrapper; ordinary ACLGraphWrapper stays unchanged and must not be reported as optimized.

## Boundaries and CPU contract

Test false-to-true-to-false transitions on the same context, backend/model replacement, compress, sparse false, metadata executor present, draft, NONE/PIECEWISE, ENPU/Eagle pre-existing exceptions, first capture, and repeated replay. Assert exact call order: preserved offloader fence then replay; only the intended target host sync disappears. The existing sparse path has no parameter-update call before or after the model. Preserve input-prep event synchronization/record and output-copy readiness protocols.

The guard is broader than the measured bucket2 workload: it does not constrain asynchronous scheduling, parallel configurations, or arbitrary custom offload implementations. Source ordering supports ordinary same-stream use, but those modes are not newly runtime-validated. Do not describe the patch as verified for every dynamic mode. If a future path mutates graph-consumed memory from another stream, it must supply a dependency or revoke this property; model/backend names alone are not an eternal ordering contract.

## Ranking and next discriminator

1. Remaining FULL execution critical path is unmeasured: eager MTP, outer input/metadata, target replay, and output dependencies. Old eager trace cannot rank their current removable budgets.
2. This host synchronization is the clearest narrowly removable framework boundary supported by present source, with unknown exposed cost and possible downstream absorption.
3. Attention launcher/tiling and local-prepare work remain concrete secondary source questions, with no larger matched removal established.

Top three questions: (1) does this host wait materially delay the next useful CPU submission? (2) are every reused CPU staging buffer and auxiliary-stream producer protected independently? (3) can the per-forward capability be revoked correctly without changing other wrappers or modes?

After CPU contract passes, the smallest runtime discriminator is the predefined single-candidate same-FULL/H6/H5 A/B/A/B comparison, all-rank transfer success, golden IDs/EOS and matching cumulative MTP workload, observers off, with the intended wrapper/guard identity proved. No profiling campaign or parameter scan is justified. Correctness failure stops the candidate; benefit absorbed elsewhere is a legitimate negative result. This review does not initiate that experiment.

Stop list: do not remove input-prep/offloader/HCCL/output-copy fences; do not extend to dense/compressed/device-metadata/draft paths; do not combine other candidates; do not clear H6/H5 or stop research for incomplete API/SLA/Current acceptance.

Most likely misread: treating stream.synchronize duration as removable device work, treating same-stream ordering as CPU pinned-buffer safety, or attributing Run265 configuration speed/old Run249 exposed gaps to H9. All are unsupported. No full Scheduler decomposition or performance claim is made.
