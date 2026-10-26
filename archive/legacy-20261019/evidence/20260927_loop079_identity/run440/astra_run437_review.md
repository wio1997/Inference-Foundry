# Run440 — independent Run437 evidence review

2026-09-27. Read-only source/evidence review and pure-CPU validation. No service command, framework import, live process/device query or NPU operation was executed. This review writes only the requested report; raw Run437 evidence is unchanged.

## Verdict

**ACCEPT Run437 as a clean scoped diagnostic, with `diagnostic_valid=true` and `all43_row_identity_certificate=false`.** It closes actual production Target FULL entry/input-storage binding and sampled CP metadata-value uncertainty left by Run403. It does not close complete all43 row identity, compulsory Resource work, physical HBM traffic, mixed capacity, executable synchronization/storage dependencies, numerical model equivalence or any finite Resource/Scheduling/Product ceiling. No numerical bound promotion is justified. No false unconditional certificate was found in the saved validator result; its explicitly conditional labels are essential.

## Independent acquisition and integrity checks

- Forty raw capture files and forty Runtime reports form five cohorts × eight ranks, each capturing cycles64/65. Cohort lengths are301,298,302,318,304. All Runtime reports pass, select FULL Target, have zero post-handoff oracle Target/ModelRunner cycles, and report12×1024 generated outputs. Independently summed capture histories equal each report's staged counts. Runtime request lists agree across ranks and contain60 unique IDs across cohorts.
- Clients report48/48 warmup plus12/12 diagnostic successes, each with exactly1024 output tokens. Independent timestamp sweep yields maximum concurrency12. The service log has exactly60 completion POST lines, all200 OK; sampled Running/Waiting pairs are12/0 and8/0. This supports the clean48+12 protocol and does not show the extra-client contamination found in Run389. Request length success is a serving gate, not a numerical-equivalence gate.
- Diagnostic bench duration21.9403498359s and560.0639958754tok/s are instrumented12-request observations, not formal Current or an overhead measurement. Warmup throughput536.4165317835 likewise is not a formal replacement.
- Re-running the current validator on copied raw captures exits0 and produces a byte-identical validation.json. Current helper/patch/validator/runner hashes exactly match final preflight v3. Independently regenerating all six patched source texts from saved originals produces every recorded patched SHA; each original backup matches its manifest SHA. Install/restore file manifests agree; before/after source manifests are byte-identical; current remote source bytes also match all six restored hashes.
- `cleanup_status.txt` records run/stop/stop_verify/restore/SHA/compare/final exit0. Saved stop process probe is empty (`[]`), and the saved final NPU probe/stop log records all eight cards at3435–3447MB and no VLLM worker listing. These certify the saved cleanup checkpoint, not an ongoing monitoring claim. The apparent “仍在占卡” line is an intermediate stop message preceding successful drain and final verification.
- Capture archive checksum matches its saved checksum file. Every one of its40 files is byte-identical to the corresponding raw capture. Helper/runner hashes are independently verified at review time against preflight; the runner did not create a separate immutable launch-time helper/library binary manifest, so these are strong consistency checks rather than proof against hypothetical intervening source replacement.

## What Resource numerator uncertainty actually closed

Run403 already supplied sampled replicated route matrices, local count parity, contiguous expert ownership, Target96×6×43 and eager Draft84×6×3 geometry, and Draft64→Target65 candidate-ID handoff. Run437 materially improves their binding to the production path:

1. For all80 selected rank-cycles, exactly one recorded FULL graph entry has the96-token/12-request uniform/no-LoRA descriptor and capture generation5. Its capture and replay explicit argument tensor descriptors match. `input_ids` and `positions` match Runtime state descriptors, including data/storage pointers, shape, stride, dtype, device and offset. Every Target router observation is tagged with that replayed entry. This closes “were these explicit state inputs bound to this selected graph entry?” for the samples.
2. Actual Target input IDs, positions, logits indices, query boundaries, sequence lengths and slot mapping are captured. All observed logits indices are exactly arange96, stronger than the validator's generic permutation requirement. Independently, every Target position equals initial slot position plus all prior masked accepted counts plus row offset. All selected slots are active, and initial Runtime output counts are0 with remaining1024. These strengthen endpoint geometry and same-cycle alignment.
3. Each selected rank-cycle contains eight updater group bindings. The five checked fields per group are `seq_lens`, `input_positions`, `start_pos`, `local_query_start_loc`, `local_seq_lens`. Saved value prefixes satisfy the source's TP8 query-block intersection/truncation formulas; local blocks are[rank×12,(rank+1)×12). “Five CP maps” means five field formulas, not five groups or a full attention metadata certificate.
4. All8 route matrices, names, ordinal coverage, ordered EP groups,32-expert ownership maps and local group counts pass. Target has43 distinct route pointers and43 distinct group-list pointers in every sample. Eager Draft snapshots are cloned immediately after dispatch on the current producing stream. Current route multiplicity and expert assignment are therefore well specified at these captured array rows.

The Resource increment is **better production provenance and endpoint/value binding of the observed partial numerator**. It is not yet an unconditional retained-token expert union, and gives no new capacity denominator.

Independently recomputed selected useful-row counts are42/60,43/40,35/43,55/64,44/38. Current Target expert-layer set cardinalities range5239–5595; conditional retained-index sets3713–4612; Draft sets224–278. If the Run405 packed W4 expert payload assumption12,582,912B/expert-layer is reused, these correspond to65.921875968–70.401392640GB current Target,46.720352256–58.032390144GB conditional retained Target, and2.818572288–3.498049536GB Draft. These are optional reproducible storage-set descriptors, **not HBM traffic or bound endpoints**. Run437 does not capture expert weight dimensions, strides, quant metadata or actual storage allocation needed to independently discharge the format assumptions. Changed ranges versus Run403 are different sampled trajectories, not an optimization or matched before/after result.

## Remaining identity, native and logits gaps

The input binding checks cover explicit graph args/kwargs, principally input_ids/positions. They do not bind every implicit forward-context attention object, model weight/cache tensor, group, captured output or native workspace. The graph key is a Python entry identity, descriptor and helper capture counter, not a native graph-handle or compiled-object manifest. Non-tensor values are represented by type only; the observed optional kwargs are None, so this does not invalidate these input checks but limits their generality.

CP checks validate metadata updater references and value prefixes, not all43 consumers. Actual per-layer attention/model/FlashComm branches, layer→CP binding, ordered attention/SP collective groups, pad/chunk/unpad layouts and output row composition are not captured. Full CP shapes/tails and every scalar/group attribute are not asserted by the validator. The saved Target slot-mapping array is cross-rank compared, not independently derived against KV ownership or every downstream write.

A common all-rank router-row permutation can preserve all route histograms and endpoint metadata while changing retained expert attribution. Run437 has no propagated per-layer row-label vector or complete native-contract proof excluding this. FlashComm/DSA/head transforms, embedding/residual path, attention/rotary, GMM row/scale association and fused collective query-order composition remain open. Native binary/tiling/loaded-object identity is also missing.

Run432v2 contributes only a sampled eager token/expert gather and mask result with pairwise slot-mutation sensitivity; scale accuracy and all-layer distributed coverage remain unproved. Run433 contributes real persistent-input-storage NPUGraph generation refresh under explicit synchronization for one96×6 fixture/range. Its tolerance-based synthetic unpermute oracle does not detect every fine slot corruption, and it bypasses the real GMM chain. Neither transfers automatically to all43 production native contracts or asynchronous replay merely because Run437 now records the actual graph entry.

Actual arange96 logits indices resolve the previously uncaptured endpoint index value, but the validator only requires a permutation. Source `target_adapter.py` indexes final hidden states with this tensor; connecting each earlier router row to that final row still requires the missing composition proof. No actual Target logits, greedy acceptance recomputation, hidden-state differential comparison or complete Draft numerical output oracle is captured. Draft64→Target65 equality proves the seven candidate IDs are handed off in active slots; Draft65 has no following-cycle check. Draft accepted-token attribution cannot identify removable body rows where other query rows can affect outputs through attention. Exact active causal flags and dependency preservation remain necessary.

Runtime reports permit a cohort/rank/trajectory join from capture slot to60 internal request IDs; saying external IDs are absent from *all* artifacts would be false. They are absent from capture JSON and the validator does not perform this join. Client JSON does not retain SSE IDs or an authoritative pre-handoff/publication membership ledger. Zero Runtime initial_output_counts is not proof of zero externally published pre-handoff tokens. `external_retention=OPEN` remains correct.

## Offline negative controls and validator interpretation

Pure-CPU mutations used fresh temporary copies of cohort1; original artifacts were untouched:

| Mutation | Actual validator result |
|---|---|
| Alter rank0 cycle64 FULL replay input_ids data pointer | REJECT: graph input tensor capture/replay mismatch |
| Alter rank0 cycle64 CP local_query_start_loc value prefix | REJECT: CP binding value mismatch: local_query_start_loc |
| Swap Target router rows0 and7 at all43 layers, identically on every rank | ACCEPT |

The first two demonstrate discrimination of the newly claimed bindings. The third is a constructive residual gap, not evidence of an observed runtime permutation: the same validator intentionally leaves all43 false. Any downstream statement replacing `conditional_retained_experts_per_layer` with an unqualified exact retained-token union would overclaim.

The validator's `diagnostic_valid` covers the raw structural checks; it does not itself read client logs, runtime pass reports, source restore, stop status, archive integrity or a model-equivalence oracle. Those additional gates were independently checked here. It is not a general-purpose full correctness certificate.

## Scheduling, storage and bound disposition

Clones are scheduled in the current stream and exported after cohort completion, with no added hot-path `.cpu()`/device-wide synchronization in the capture helper. This supports the intended low-intrusion design but does not prove zero overhead or every internal producer-stream join. Snapshot copies and graph hooks can perturb submission, memory lifetime, cache traffic and execution. No uninstrumented A/A control is present.

Distinct Target pointers and successful retained snapshots are sampled lifetime observations. They do not prove all graph allocator reuse, native workspace lifetime, side-stream count-copy completion before overwrite, all-rank collective readiness/join, cross-rank clock alignment or an executable reordered schedule. Run401's count-copy/storage proof obligation and the original-path dependency closure remain separate. Run432/433's synchronized fixture cannot settle these asynchronous obligations.

Keep all finite Algorithm/Hardware Resource, Scheduling and Product ceiling fields null. Preserve formal Current Run99 median571.681tok/s. Next gates should target the residual layer/native row composition and actual metadata consumer bindings, plus expert-format provenance for the Resource numerator; separately settle count-copy/storage and original graph/collective joins with controlled overhead. Complete compulsory compute/traffic and credible mixed-capacity upper bounds are still required before division can yield a defensible latency relaxation.

## Verified SHA256

- capture.tar.gz: `3efd868938db55658e927ed0a76aea6d588cdab9ff4238ece26aede110579dd3`
- validation.json: `fcd5acd45b423bd3a92c2440f3d838ed0a7efac8d92ab9cd58e73c1867c3f292`
- bench.json: `24fb16bc7eac7f4743459e0ba73a7babd80aa52780ebdced787ccd3b07799698`
- warmup48.json: `1d26d0ec03f3d30ff81e90c6c81c46bf1f5476e388f78babf1680ad7a70d46bd`
- cleanup_status.txt: `c25d5b05820a0950b825f88ba595e1be0477f758aef47a75e01cd776c1b1d7d4`
- source_before.sha256 = source_after.sha256: `e06aa6a98b8d3816af5ca051d518d8ee981ca0443f1f2801e9d6ff34e7ed6618`
- raw service log: `a6d3635d7b298d5cb6afa8d465597b46660dfa3cd69070d39c3d5941a6d81594` (108651 bytes)
- loop079_row_capture.py: `632d87dfbb1b84109eb0740c5bdca8446fa41ec075b2c55453faec5ad12de377`
- loop079_row_capture_patch.py: `6edb6247c1de807a4ea229ac17fa68b99e0ce4bf2beec5e26fa357028c47e103`
- loop079_row_capture_validate.py: `0d6f67f1ed879cbbf6419774402069a2f43893a702f84fe366fd24dac7181381`
- run_loop079_row_capture_run437.sh: `ab845a613ce44f61c165d2a6fd5b060885101e83723cb843478bf81b38f24589`
