# Independent MoE gather diff review — 2026-10-06

**No blocking correctness defect found for the stated GLM-5.3 equal-shard, hidden-width6144 path. KEEP the candidate for correctness validation; not deployment or PERF_KEEP.** Reviewed moe_gather.patch, original prepare/finalize/MC2 consumers, and supplied ProcessGroupHCCL.cpp at installed torch_npu git_version5dd8ef3f9b375b5ae4a83538d5785754148c3302. Build-commit provenance is supplied by the controller; this review personally verifies the corresponding downloaded source behavior, not an independently rebuilt binary.

## Diff assessment

- Branch uses positive padded row count divisible by TP, a common group shape under existing prepare/finalize invariants. Local input-shape check asserts rather than choosing another collective, avoiding the prior rank-local fallback hazard. A violated invariant will fail the operation; it does not authorize recovery by retrying a different collective. Existing code likewise requires consistent group geometry.
- The asserted input shape plus fresh output shape gives rank-major concatenation with output numel=TP×input numel. Same explicit process group and same post-gather unpadding are retained. MC2 inherits this finalize except its unchanged replace_allreduce branch. TP1, replace_allreduce, zero-row/nondivisible original branches remain unchanged.
- `.contiguous()` consistently normalizes the input, preserves values and does not mutate the original. It may allocate for a noncontiguous input; this is a correctness-preserving conditional cost, not assumed free.
- Fresh `torch.empty` output is preserved. No prepare-view alias, persistent-buffer reuse, earlier publication, collective removal or stream change is introduced. Returned slicing retains the backing output storage.

## Exact backend closure

ProcessGroupHCCL::allgather at5572 casts inputs to origin format, flattens gathered outputs, invokes HcclAllGather, then its post-callback iterates output slices and performs `copy_` (5637–5665). This directly explains the observed TP16 output-copy pattern.

`_allgather_base` at5895 checks dtype and output/input numel relation, validates NPU tensors, casts the input to origin format, and invokes the same HcclAllGather directly into the provided output, with empty pre/post callbacks. Thus the source supports removal of the16 output unpack copies, without claiming a different collective algorithm or kernel implementation.

Lifetime: common collective code around4198–4260 records input storage on the HCCL stream or stashes it in AVOID_RECORD_STREAM mode. The base callback records output storage, and common code stashes output for AVOID_RECORD_STREAM; work also owns outputs. This covers a temporary `.contiguous()` input and the fresh output through asynchronous backend execution. The patch retains the ordinary synchronous Python API choice; backend stream dependencies remain responsible for device completion.

Format: both paths call cast_to_origin_format, which preserves base-format inputs and converts nonbase inputs to recorded origin format (3351). Fresh torch.empty output is ordinary dense storage. Logical contiguity is therefore not the sole format argument; the inspected backend supplies the additional normalization.

## One bounded caveat: alignment

The list path's byte_alignment (5304,5595) pads per-rank byte counts to512B on pre-Ascend950 hardware, then slices after gathering. The base path does not perform that padding. For the stated GLM hidden width6144 and BF16/FP16/FP32, every integer-row shard has a512B-multiple payload, so no alignment difference arises. Do not generalize the proof to arbitrary hidden widths or packed dtypes. If such shapes are in the intended patch scope, preserve them with a common-shape byte-alignment guard or verify the actual HCCL nonaligned behavior; CPU Gloo cannot resolve it. This is not evidence of a bug in the stated GLM path.

## Validation status and limits

The four-rank Gloo job was running when reviewed; no pass is claimed here. Its planned dynamic rows/dtypes/noncontiguous/input-unchanged/fresh-buffer cases are meaningful for value/layout and ownership semantics. Confirm tests execute the actual patched method and separately exercise original nondivisible/fallback branches. Gloo does not prove NPU format, allocator-stream behavior, graph behavior or backend copy removal. A separately authorized HCCL correctness fixture and short trace remain the smallest backend check before matched profiling-off complete PD evaluation. Inclusive209.7ms cost and profiled host gaps are not predicted E2E savings.

No files other than this review were changed; no service, model request or NPU operation was performed.

## Run250 gate failure review — 2026-10-06

Personally read runs/GLM-RUN-0250/failure_raw_excerpt.txt, prepare_finalize_shim.py, original/candidate finalize consumers and routed_experts/token_dispatcher padding-mask handoff. **PARK performance adjudication; keep the candidate undecided while repairing the correctness diagnostic.** No independent server action.

Facts: rank12 raised at the combined `assert equal and torch.equal(before, hidden_states)`. The shim writes its witness only after that assertion, so both rank12 output-equality and input-preservation outcomes are unknown. The15 other witnesses prove their own retained output and numeric input-equality checks passed; they do not determine which rank12 predicate failed. A rank-local exception before peers' subsequent model collectives can strand those peers, consistent with the observed broadcast warnings.

The gate has two concrete weaknesses:

1. `torch.equal` is numeric equality, not bitwise equality. Matching NaNs can make it false; signed zeros can compare equal despite different bits. The field `actual_hidden_bitwise_equal` therefore overstates what was tested. Identical NaN-bearing input bytes could explain the preservation failure, but no recorded tensor values/masks prove that occurred on rank12. Do not mark the failure INVALID-as-false-negative on that hypothesis alone.
2. Both compared finalize methods already unpad. With num_tokens1/padded16 and one row per rank, expected/result compare only retained row0. This is a useful public-result check, but does not certify all16 gathered shards or rank-order placement. A full-layout diagnostic must inspect the full gathered buffer before unpadding.

Padding interpretation: MC2 passes its active mask through routed_experts into dispatch/combine; the gather is concatenation, not arithmetic reduction. If the stated one-valid-row geometry is confirmed, rank12's row is discarded by `[:1]`. Thus a NaN in that unused local row can be numerically irrelevant to this finalize's retained output while still breaking torch.equal(before,input). This does not prove the padding kernel must produce NaNs, prove bytes were unchanged, or justify tolerating NaNs in real rows. The mask/kernel's unspecified inactive-output values must not be promoted into a diagnosed cause without data.

Bounded next diagnostic: use the independently owned16-rank HCCL fixture before model reload. Compare (a) full gathered bytes against rank-coded expected concatenation, (b) post-unpad bytes, (c) each input's before/after bytes, (d) separate finite/NaN-mask diagnostics. Include explicit padded NaN/Inf/+0/-0 payloads and finite valid rows, equal/unaligned-fallback geometry, noncontiguous input, fresh-buffer preservation. Byte comparison should use integer/byte representation without numeric conversion; equal_nan allclose is not a replacement for exact-copy verification. Retain individual predicate results, counts/first differing byte and shape/dtype/rank before deciding the verdict. Do not log only the combined Boolean.

All ranks must complete the same planned collectives and exchange diagnostic status before any local assertion/return that allows another forward to start. This collective status gate handles a value-test failure; it cannot rescue a failed/hung HCCL operation itself, which needs the bounded controller timeout/teardown. Do not hide the mismatch by zeroing live padded inputs, ignoring rank12, weakening tolerances or removing the input-preservation check.

Reported short8 output matching and lower B1 timing are provisional local evidence. The complete math outputs differ in work (A1/A2=74 tokens versus B1=8 despite final answer2), so their complete times are not a matched fixed-work performance gain. They also deserve investigation independent of this diagnostic NaN hypothesis. No performance KEEP/Current change follows from this review, and source restoration on disk does not prove a hung resident worker has reverted its in-memory shim.

## Run251 and final critical-path review — 2026-10-06

Personally read Run251 comparison_results.json, B1/B2_witness.json, micro_checks/rank7.json, execution_summary.json, the actual shim, and decode_path CRITICAL_PATH.md, step_timeline.json, reduce_compare.py and SFA resolver. No server/NPU action. No blocking evidence overclaim found in the current CRITICAL_PATH verdict; retain Current=None and complete E2E INCONCLUSIVE.

- NaN conclusion is properly bounded. The shim compares uint8 representations, temporarily retains all padded rows for both gathers, restores num_tokens, and exchanges all-rank status before its value-test failure. All32 live witnesses pass full-output/input-byte checks; B1 rank7 records22 NaNs and B2 ranks14/15 record24 each. Rank7's explicit special-value fixtures demonstrate numeric self-equality failing while bytes pass. This establishes why the old gate was unsuitable, not which Run250 rank12 predicate failed. It does not license NaNs in valid output rows.
- The short8-token results match token IDs/chunks and show a positive scoped signal: median TPOT improves6.42% and10.32%; all four B samples are below all four A samples. Two samples per phase do not establish an SLA or stable distribution. Complete3-token requests also match IDs/chunks/natural EOS, but each phase has only one complete sample. First-pair improvement33.15ms is smaller than the52.33ms spread between A phases. B2's179.48ms whole-PD improvement includes31.16ms faster P time, which this D-only patch cannot directly explain. Thus the whole10.28% must not be attributed entirely to the gather change. Current INCONCLUSIVE adjudication is appropriate; no additional Run is requested by this review.
- Actual SFA resolver independently selects AscendSFADCPImpl when replicated-indexer DCP is enabled and DSA-CP disabled. CRITICAL_PATH correctly follows SFA compact-KV/packed-Q gathering and sfa_dcp_a2a_fused. Earlier conditional references in my decode review to legacy dsa_cp head-layout communication are not attribution of the active path and are superseded by this correction; DSv4 dsa_v1 is not established as active.
- Union, event-wait and peer-wait descriptions retain overlap and profiler-regime qualifications. Late host submission supports a host-supply bottleneck; it does not make all exposed gaps removable Python time. Neither the step timeline nor current write-up converts collective-union/event-wait duration into additive savings or a decomposition of the profiling-off265ms/token.
- Reducer scope: reduce_compare.py checks source/method identity, witness and summarized token/chunk invariants; its INCONCLUSIVE verdict is explicitly assigned. It is not an independent reconstruction of request bodies, KV-hit deltas or timing from raw transport. Keep those claims grounded in their original evidence rather than describing this reducer alone as a complete contract validator. execution_summary correctly marks full_API_contract=false and formalSLA=false.
