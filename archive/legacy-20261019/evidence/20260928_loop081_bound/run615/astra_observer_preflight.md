# Run615 — minimal reversible typed-KV observer preflight

**SCOPED IMPLEMENTATION DESIGN PASS; NOT LIVE-READY.** Independent read-only Astra High review of Run613/614/615 and installed source. No new NPU work, service or live patch. Run615 proves device0 descriptor availability and original-stream copy-before-overwrite behavior on a toy ND allocation only. Its 776-byte scratch accounting is not the real observer's budget or a cache lifetime certificate.

## Smallest useful implementation

Build one root-owned observer module plus narrow reversible callsites at (a) DSpark context projection/store, (b) the actual query SWA scatter, and (c) the actual SparseAttnSharedkv invocation. Gate by explicit measured cohort5/logical cycles64–65, layer43–45 and actual eager dispatch; use existing boundary lineage for predecessor/next Target. Default recorder is absent/no-op. No sampling decision may depend on a copied device value during execution.

For the observed unsplit path, expect 18 role records/rank: two cycles × three layers × context-write/query-write/attention. Actual prefill/decode splitting may add records: predeclare per-branch subrecord caps and offsets, validate the observed dispatch, and report unsupported geometry rather than silently forcing 18. Never monkey-patch a global op in a way that changes Target Graph capture/replay. Compile/selftest the concrete changed callsites and verify exactly one hook installation; Run613 nested same-label hooks warn against accumulating observers across cohorts.

One immutable record carries run/rank/cohort/logical-cycle/request-row map/layer/group/role/branch/ordinal/submission-token. Label the existing custom-op call with a uniquely named record_function scope. Match its contained CPU op on same pid/tid and full-duration containment, then exact torch_to_npu s/f ID to native task, as in Run613. Save native model/stream/task/batch/subtask identity. A label, current ContextVar, connection_id or Host timestamp alone is insufficient. ContextVar is set/reset in try/finally; freeze the record before enqueue. Unsupported Graph-only calls remain unobserved, never fabricated as eager witnesses.

## Scheduling witness admission S

**S1 lifetime and physical view.** At cache binding register the Runtime-owned persistent storage owner, cache group/layer, device, storage base, _cdata, nbytes, logical offset/shape/stride/dtype and actual NPU format. A session-local monotonically assigned token names that owner's lifetime; _cdata by itself can be recycled. Require the original Runtime to hold the cache throughout the window. A diagnostic strong reference must not silently extend a normally releasable production allocation and thereby erase a reuse hazard. Dynamic lifetimes need allocation/free or ownership-transition instrumentation; otherwise mark them UNKNOWN. Distinct tensor objects sharing storage must resolve to byte-range aliases. Do not assume Python shape/stride gives physical byte mapping for a padded/device format; admit ND only until another mapping is proved.

**S2 content and writer epochs.** Allocate separate tokens for storage lifetime, metadata snapshot content, and each production scatter invocation. Derive per-destination-row last writer using the actual formatted slot payload, block size, request/position and valid/padded mask. A single scatter with duplicate destinations is ambiguous unless the loaded kernel's duplicate-write semantics is established; do not choose the last input index by assumption. Existing retained rows begin as ENTRY_UNKNOWN unless predecessor provenance was captured. Keep query overwrites and valid reuse as observations.

**S3 ordered snapshots.** Preallocate all destination tensors before the selected workload. Snapshot mutable slot maps and consumer metadata on the existing stream after producer and before reuse. Reuse one snapshot only when its content epoch is demonstrably shared. A Host descriptor alone does not establish device readiness. Missing cross-stream ordering remains UNKNOWN; add no wait/event synchronization to repair it. Export after the existing drain. No .item(), .cpu(), device-value formatting, synchronization or per-record allocation is allowed in the active window. Snapshot-copy tasks must be separately labelled and excluded from production task/traffic counts; they still perturb resource timing.

**S4 eligible reader.** At the real attention call capture actual cache/view, branch, q row mapping, cu_seqlens_q, seqused_kv, block table, block size, mask mode, windows, layout, head dimensions, sparse indices and metadata content epoch, including None. dsa_v1.py:1239 onward uses the actual SWA block-size kwarg rather than a universal128; prefill and decode builders differ, and noncausal decode may use explicit dspark_swa_indices and another window. Inspect the invoked loaded op/tiling/ABI before deriving eligible rows. A CPU reference must map semantic query rows to cache byte ranges without using source-shape guesses. ABI eligibility is not a physical load timestamp, compulsory HBM crossing, or proof of nonzero numerical influence. If the selected kernel's row semantics is unavailable, retain exact identity but mark reader edges UNKNOWN.

**S5 lineage closure.** Join each eligible row to the latest preceding write under proved stream/event order, covering intervening aliases and predecessor generations. “First reader” means first within the covered instrumented reader set; do not claim globally first without coverage. Preserve Draft-output/state→next Target-input identity: Target does not read Draft SWA directly. A current storage RAW/WAW hazard does not prove universal seriality; alternate versioning/layout/recomputation may eliminate storage ordering while retaining semantic dependence.

S can PASS scoped typed identity even if some edges remain explicitly UNKNOWN; only fully covered edges may enter the current DAG. Do not label an incomplete DAG a Scheduling lower-bound certificate.

## Independent fresh-work admission F

Evaluate F only after S identifies one unoverwritten eligible context row. Declare the online fixed-expression arithmetic class; pin actual input semantic key, Target-aux/combine producer, weights/quantization, ownership and selected output. Copy bounded selected operands without Host-dependent branching. Audit permitted pre-entry result stores and semantic duplicates. Same request/position, new pointer, or changed hash alone does not establish fresh necessity. An incomplete store inventory or missing successor yields UNKNOWN; legal reuse, overwrite or masked row yields NO_WITNESS for that candidate while S stays usable. Do not multiply replicated work by8 without ownership proof. F can admit a conditional arithmetic subset, never compulsory HBM or a positive time floor by itself.

## Executable preflight gate before live

Require: exact patch/source/binary SHA manifest; concrete schema/reducer; numeric record cap and total metadata bytes/rank/all8; selected operand scratch ≤128KiB/rank; overflow is explicit rejection, never truncation. Calculate budgets from frozen actual dimensions before install. Include allocation lifetime and metadata alias fixtures, not just Run615's toy descriptor test.

Run CPU positive/negative fixtures for known writer→reader, pointer reuse, same storage/different views, slot content overwrite, duplicate destinations, invalid/padded/parked rows, stale block table, branch/block-size mismatch, unknown native format, missing alias writer, delayed device work across next Host Target, missing/duplicate flow, exception-scope reset, missing predecessor/successor, overflow, equivalent pre-entry/cross-request result, same position/different prefix, wrong quant and false replication. Assert unsupported claims fail while legal reuse/overwrite remains a valid Scheduling observation.

Audit the patch's active-window operations to allow only metadata descriptors, preallocated copies and CPU scope records; model/collective/synchronization counts and numerical inputs must remain unchanged. Define same-state acceptance/count/KV/state/output comparator and self-replay nondeterminism policy before the guarded acquisition. Independent-run token hash difference is not automatically an observer bug. Preserve client/Basis/Product/dispatch admission and guarded tagged stop/source restoration.

Identity-only admission needs no new A0/A1 service run. Any timing transfer, legal-overlap service model or claimed Runtime gain requires observer OFF/ON/OFF and then controlled schedule intervention/formal E2E as appropriate. Strict Resource/Hardware, Scheduling/Execution and Product endpoints plus numeric gap remain null; formal Current571.681 tok/s.

## Input identities

/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run613/eager_flow_join.json
SHA256 dc17ab40f85d38d2d5f87d921d44d1b1f4debb51dea6aceddd9b9ae0ad9c99b9

/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run614/typed_kv_packet_gate.json
SHA256 73ed975c9bab9eb47ea877741f223e62e9eda1de94c56d76010ef259c3d85f79

/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run614/astra_packet_final_check.md
SHA256 32d85f4928c5e2346ba0d63b379adca8db8ae127a8ff7e69a3e7f0d54ad638af

/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run615/snapshot_api_smoke.json
SHA256 902439f0fe5fe46079504b5d9e53b3ff49ef26d7f6df4e6a7a3247490aa198d2

/data/wio/Inference_Foundry/scripts/loop081_typed_snapshot_api_run615.py
SHA256 070e1805a3cb964b49d19bb1a5dc1775b60a905f3235a644b85ca19fa1d6c6f6

/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py
SHA256 b459fa373e89da1722085cffa63a9959502dd29fb1a0cf4b8f073e431d811867

/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py
SHA256 e9163996db794c5fba777ea55a5374283780a2fd3404778f5369caea76f9344f

/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py
SHA256 e3e1ff579e67f845f155c983294ae5546e6ee76ba330098d7867e870c3f9f6e4

/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/dsa_v1.py
SHA256 88a0cd69dbbba476d0ac179431568590e6e876134390fba77f22721069f491cb

/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/device/device_op.py
SHA256 67dda24f28de847f9aee87db9714e5d2293ebb46309ad13152538cf63ca5f1b4

/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/dsa.py
SHA256 3abeb11a3d82f3919d67f4f7c5ab404faa476c2eb8455d2eca777dc1712811fc

