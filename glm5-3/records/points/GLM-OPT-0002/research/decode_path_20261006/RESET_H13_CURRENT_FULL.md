# Current-FULL H13 research reset and Goal Review

2026-10-07, after Run280 completed. Source/profile research only at this reset; no Run281, candidate device installation, new profile or parameter scan. Preserve H6/H5 and all existing correctness/KV/API repairs. H11/H12 off. FullAPI/Current/SLA unfinished does not end research or discard validated patches.

## Five-question Goal Review

1. Product: GLM-5.3 W8A8 standard P Prefill→KV transfer→D Decode/MTP, dynamic valid requests and full external API; formal input80K/output600/repeated prefix and declared KV-hit93%, with isolated round namespaces and measured per-endpoint cache counters.
2. Time: old eager265ms/token is historical, not the current target-FULL budget. Existing Run276 raw places steady profiled step around55–56ms, ~52.3–52.4ms nonwait device union and largest inspected gap137us. CPU sync/output waits overlap devices; no evidence for deleting them as additional35–40ms. H11 merged MTP graph correctness passes but matched Run275 has no repeated D gain. H12 cached-RoPE materialization correctness passes Run279; Run280 work-confounded/no repeated gain, so off.
3. Largest removable gap: global maximum and saving remain unknown. Next concrete source-proven redundant region is target MC2 global padding followed by TP local slice. Existing rank0/13/15 period3 CSV has75 hidden [2,6144]→[16,6144] pads,75 router [2,256]→[16,256] pads and150 MemSets in captured Model49/Stream93 per rank. Whole Pad/MemSet family includes one eager MTP pair and totals3.66–4.05ms task-inclusive. This is neither additive critical-path saving nor a gain prediction. Current captures still perform16-row materialization although each TP16 rank consumes one row; fourteen peers receive empty-input zero shards.
4. Recent experiments: H11/H12 close legality and exclude unconditional gain claims; they did not establish a new performance patch. Retain their sources/raw/correctness and stop unchanged toggles/repeats. Local prepare was previously CPU-only parked; current FULL task geometry gives a new concrete device-cost reason to revisit it.
5. Restart decision: continue one H13 question. Patch local preparation, preserve necessary TP/DCP/EP/MTP communication and shared-expert/input ownership. Do not branch into new kernels, versions, parameter scans, generic runtime rewrites or blind repeated reloads. Establish honest graph capture comparison first, then one bounded correctness diagnostic only if it can decide admission to matched A/B/A/B and complete PD E2E.

## Hypothesis and code

For non-SP padded length divisible by TP, replace global padded hidden/router tensors with the exact rank-local intersection; clone full valid shards, pad partial local shards, allocate **fresh zeros** for empty shards. Preserve global padded_hidden_states_shape, num_tokens, masks, token order, collective participation, fallbacks, routing and finalize. No cached immutable zero tensor or borrowed caller-input alias.

Production candidate `post_full_local_prepare/prepare_finalize_candidate.py` SHA `7bb35edfe568bb62b3432c3406e0a66d23202cd1608bebcb1ce1016789d73dbb`; original54e8 unchanged. New actual CPU3888 passes bitwise all-rank reconstruction, masks/strides/input isolation, noncontiguous/unequal/zero/NaN/Inf/signed-zero and SP cases, no NPU initialization. CPU result is not device-format/lifetime/performance proof.

Pinned op-plugin8b9c8534 op-api ConstantPadNd source has no empty-input fastpath; legacy aclops has one but is not the observed aclnn path. Explicit fresh new_zeros avoids assuming empty F.pad skips its pad kernel. Existing correct fill/clone/pad kernels are reused; actual capture and format still need hardware proof.

## Decisive missing evidence and gates

Python prepare mode alone cannot replace the75 layers already captured in target FULL. Actual wrapper owns one entry per BatchDescriptor; an honest comparison must capture both implementations. Source follow-up found no prohibition on two same-model wrappers with independent retained pools, and current SFA has no task-update consumer conflict. Actual workspace d0650393 repeated lock is idempotent; equal/smaller requests keep addresses, growth raises. Never unlock/reset it to force a second capture.

Actual-AST CPU lifecycle oracle passes distinct pools/entries/private outputs and A/B/A/B, NONE eager, workspace and preserved Ascend contexts/sync/offloader. It uses device/graph/tensor/weakref and dummy-model doubles: it does **not** prove NPU allocator, HCCL, actual metadata/KV or async-consumer safety. Idle/invalid gates are proposed controller requirements, not upstream guarantees. Upstream capture failure can leave capture-enabled=true; abort and use only the declared bounded recovery, never continue requests.

Before any device action: review a concrete narrow research-only bank shim and test its actual wiring; freeze sources, fixed work, cache salts, observation-off measurements, all16 capture/replay witnesses, controller uniqueness and one recovery. Baseline and candidate must retain same descriptors, raw weights, metadata/input pointers, H6/H5 and all sync fences; independent graph-private pools/outputs stay owned. No graph-param reinitialization, graph clear, silent recapture or eager fallback counted as candidate replay.

| Evidence | Decision |
| --- | --- |
| Source/shim CPU or bank ownership/capture contract fails | Fix offline; no NPU Run |
| Bounded native exact output/byte/layout/reset/actual all16 replay fails | Preserve failed raw; one bounded H6/H5/FULL recovery; no A/B |
| Correctness and actual old/new banks pass | Freeze same-worker matched A/B/A/B plus natural complete PD, heavy observation off |
| Work/cache differs, improvements fail frozen noise/repeatability gates or PD gain absent | INCONCLUSIVE/REJECT as appropriate; do not promote/retry for favorable values |
| Correctness, admissible repeated D and complete PD gains pass | Add H13 to active research stack; continue next optimization on H6/H5/H13; fullAPI/SLA/stability decide final Current separately |

Independent reviews: `POST_FULL_LOCAL_PREPARE_REVIEW.md`, `LOCAL_PREPARE_DUAL_BANK_LIFECYCLE_REVIEW.md`, `post_full_bank_sources/LIFECYCLE_ORACLE_REVIEW.md`. Root personally read actual source, diffs, raw reductions, geometry and CPU oracle. Present state: one H13 source hypothesis, one candidate bank diagnostic under preparation; no device result or gain.
