# checkpoint180 — DeepSeek graph comparison closed; current-FULL local preparation

2026-10-07. **本阶段没有新增代码级性能 KEEP。** Active research stack remains H6/H5; target FULL retained. H11 and H12 off. Final Current/fullAPI/formal80K input/600 output/93% prefix and declared KV-hit SLA remain open; research continues.

## Results and scope

* DeepSeek reference is accepted: ordinary MTP supports merged draft-model/logits/sample graph capture. GLM original has a specific eager override; H11 removed it with GLM SFA/DCP lifetime fixes and already passed all16 actual replay/exact short and natural EOS PD in Run273. Same-worker matched Run275 did not repeat D improvement. The K1 draft-prefill naming in V2 is still D-side draft work. DSpark's separate eager restriction is not a general MTP restriction. Source/cost review retained; no unchanged graph toggle is rerun.
* [Run279](../../runs/GLM-RUN-0279/summary.md) COMPLETED/correctness PASS, four fixed real PD requests/no recovery. All16 old/dense cached RoPE layouts and persistent output pointers, exact IDs/usage/EOS/SSE and native KV transfer pass. Target and ordinary MTP both cached. Original Run277 FAILED and Run278 parked/not started are unchanged.
* [Run280](../../runs/GLM-RUN-0280/summary.md) COMPLETED, ten fixed same-worker requests/no recovery/reload. Raw integrity passes, but B2 has12 drafts versus11 in the other complete requests, and D savings +22.649/−38.227ms are not repeated. Independent reduction records measurement/comparison inadmissible and INCONCLUSIVE. H12 stays off; no work normalization or favorable retry.

At Run280 terminal: P249 root1916718/start303049910, D279 root3579883/start311887281, all16 healthy/idle; H6/H5/mainFULL retained, H11/H12off, H12 transition9. These are the last observed terminal identities, not authorization to act on stale PIDs. No active device controller/Run281 has been queued at this checkpoint.

## Remaining critical-path question

Old eager265ms/token is not the current-FULL budget or code gain. Existing Run276 trace/CSV is reused without a new profile or parser replay. Current steady profiled step≈55–56ms; nonwait device union≈52.3–52.4ms and maximum inspected gap137us. CPU sync/output waiting overlaps that device work. Largest overall code-removable gap and savings remain unknown.

New CPU-only task/CSV join under `jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007/mc2_pad_geometry` preserves raw hashes and912 rows across ranks0/13/15. Per captured target step/rank:75 hidden [2,6144]→[16,6144] BF16 pads,75 router [2,256]→[16,256] FLOAT pads,150 MemSets, Model49/Stream93. One extra pair belongs to eager MTP. Family inclusive totals≈4.050/3.739/3.662ms are observed costs, not predicted savings. Actual54e8 prepare globally pads before TP16 selection: every rank computes16 rows while retaining only one; fourteen ranks consume zero shards. This is the next concrete redundant framework preparation region.

Production H13 candidate directly materializes the exact local slice, clones full valid rows to preserve caller-input isolation, pads partial rows and creates **fresh zeros** for empty rows. Global finalize shape/num_tokens/masks/routing/collectives/fallbacks/SP remain. No immutable zero reuse, kernel change or skipped peer communication. New actual CPU3888 PASS covers bytewise all-rank reconstruction, empty/unequal/noncontiguous/NaN/Inf/signed-zero, strides/masks/input ownership/SP and retained outputs. No NPU/CUDA initialization. Candidate7bb35e/source54e8 is not installed or device-proven.

## Honest capture comparison

Python mode changes cannot replace the target's captured75-layer preparation. Root read actual wrapper/base/runner/workspace and independent reviews. Two wrapper objects with retained independent pools and one raw model are source-permitted; current SFA has no identified task-update cross-bank conflict. Actual workspace lock is idempotent; equal/smaller requests keep its pointer, growth fails. Do not unlock/reset or reinitialize graph params.

Actual-AST CPU lifecycle oracle passes two pools/private entries and outputs, A/B/A/B, NONE eager, workspace and preserved Ascend contexts/sync/offloader. Device/tensor/weakref/dummy-model operations are doubles; native allocator/HCCL/metadata/async-consumer correctness remains unproved. Capture exception leaves the upstream capture-enable flag true: experiment must abort and recover through its one declared path. Idle/invalid-lifecycle gates are proposed controller contracts, not existing upstream guarantees.

Unique next question and decision gates are frozen in [H13 Reset / five-question Goal Review](RESET_H13_CURRENT_FULL.md). A narrow research-only bank shim is under preparation; concrete source/wiring review and CPU proof precede any one bounded device correctness diagnostic. No generic runtime bank product, large E2E or parameter scan is authorized by this checkpoint alone. After correctness, require admissible matched A/B/A/B and complete PD E2E before H13 joins research stack. Keep H6/H5 throughout and continue on any newly validated stack.

All previous failure/raw/frozen-source records are preserved. Current tiny fixtures use fresh local-cache namespaces and actual per-endpoint counters; D external100% transfer reuse is not called formal93% prefix hit. AISBench80K/600/.93 and round-isolation updates are already present; corresponding remote branch files match local contents. Git sync preserves both histories without force.
