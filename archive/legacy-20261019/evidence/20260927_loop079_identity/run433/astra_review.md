# Run433 independent graph replay review — Astra

2026-09-27. Source/result/log review only. No torch import, NPU operation, service action, or borrowed-source modification.

## Verdict

**The script genuinely captures one NPUGraph and replays it against mutated contents of persistent input tensors.** Evidence supports a sampled synchronized fixed-storage MoE graph replay contract for96×6/H4096/E256, quant_mode1, row_idx_type0, active range[0,32). It does not certify vLLM's actual graph entry/bucket, asynchronous scheduling, all43 layers, or arbitrary exact unpermute slot sensitivity.

## Why this is real fixed-input-storage replay

`x`, `ids`, `probs`, and `payload` are allocated once outside `stage`, `forward`, and the replay loop. `stage` mutates x/ids/probs with `copy_`; it does not reassign these tensors. Host-to-device temporary tensors supply copy values, not new tensors bound into the captured forward. Payload remains constant throughout.

The script warms up eager once, creates a single `torch.npu.NPUGraph`, captures one `forward()` in its context, retains `graph_output`, then calls `graph.replay()` exactly once per generation. It does not rerun Python `forward()` inside the replay loop. `validate` reads retained capture output tensors each time. Every stage and replay is followed by device synchronization, so validation reads the completed generation under these deliberately serialized conditions.

This establishes fixed-storage reuse from Python source/lifetimes. No pointer/storage/graph-handle manifest is recorded, so result.json alone cannot independently audit the addresses or runtime graph identity. The strong but bounded statement is “the reviewed script uses persistent tensor references and in-place copies,” not “the production vLLM graph pointers are certified.”

## Observed generations and validation

| Generation | Expert IDs / marker | Local routes | Index digest | Max error |
|---|---|---:|---|---:|
|1|Run403 fixture / shift0|85|15ec8495…|0.00048828125|
|2|each expert shifted17 modulo256 / marker shift1|69|ec5e7348…|0.0001220703125|
|3|original fixture / shift0|85|15ec8495…|0.00048828125|

Actual assertions require each valid index inside the corresponding expert-count prefix segment and unique across the local prefix, each excluded index-1 and its staged probability0, the packed INT8 row's maximum marker coordinate matching the current shifted token label, exact expert histograms, and unpermute agreement with a CPU sum from current captured indices plus payload/probabilities. Index digest1=3≠2 is explicitly asserted.

Consequently this is stronger than testing only changed input IDs or a changed checksum: stale generation1 packed markers cannot satisfy generation2's shift1 check, and stale counts cannot satisfy69 versus85. Restoring generation3 shows the buffers can be reused after the changed-input replay. Marker_shift changes independently of expert assignment, helping distinguish stale packed data from merely changed index metadata.

## Limits of the native-chain claim

1. `out` consumes fixed synthetic BF16 payload and graph-produced `abs(indices)`/staged probs; it does not consume packed INT8 routing output/scales through GMM. This correctly isolates index use but does not test GMM or scale row pairing. Scales are neither validated nor recorded.
2. The CPU expected unpermute sum uses graph-returned g; token+expert checks independently constrain g, but there is no saved full g/q/probs/out snapshot to replay the audit offline. Digest is a checksum of Python list string representation, not a canonical tensor byte schema.
3. The graph payload is `(packed_position+1)/128` after BF16 conversion and comparison threshold0.015. This is weaker for some fine slot mutations than Run432v2's expanded k signatures. Neighboring packed values can differ by0.0078125 (or round equal at higher indices); weighted differences can easily be below0.015. Thus Run432's1440 k-pair mutation sensitivity does **not** transfer to this graph oracle. This graph run rejects gross stale generations and verifies sampled numeric agreement, not every possible wrong gather address. A future exact per-k channel oracle or CPU mutation sensitivity test should be used before claiming all-slot graph ABI closure.
4. Input probabilities are read back from the device for the expected sum, with excluded slots checked0; valid slots are not separately checked against the intended host powers-of-two. That leaves probability-copy corruption partially self-referential. There is no input/output pointer ledger or assertion-enable manifest.
5. Three serialized replays exercise one graph, one shape, one partial range and one route fixture with a deterministic transform. No alternative graph bucket, nonzero range, multi-stream race, event ordering, overlapping replay, multiple processes/ranks, or vLLM graph capture/replay dispatch is covered.
6. No compiled-object/loaded-library provenance is included beyond torch_npu version. Source review-time hashes below do not certify exact executed binaries or prior script identity.

## Permitted promotion and next gate

`MOE_GRAPH_SAMPLED_FIXED_STORAGE_GENERATION_REFRESH_OBSERVED` is justified for this script/configuration: under synchronized replay, changed IDs and row markers update counts, gather mapping and packed token markers, then restore the initial result. The graph-index-to-unpermute chain passes the stated sampled tolerance.

Keep the larger certificate conditional. To bind this to Run424, still need actual production wrapper+entry+batch descriptor and capture generation, input kwargs storage identities, per-layer branch/group/EP-map metadata, selected-cycle snapshots, and all43 coverage. Other attention/rotary/GMM/collective native contracts and external request retention remain separate prerequisites. This run does not warrant a new full-model measurement by itself.

Low-cost improvement before any broader graph claim: preserve current results; improve exact channel oracle and test mutations purely on CPU, record graph/input/output storage pointers and full small index/count/output artifacts at cohort end in any already planned replay. No additional NPU execution was performed for this review.

## Review-time SHA256

- `scripts/loop079_moe_row_graph.py`: `d79d6c144d4e5d730215057dd3d264b5ea7430dd609ba55748482f080287ba53`
- `evidence/20260927_loop079_identity/run433/result.json`: `8c1072edec4aee1cc7d2d7ed79565ce91eb16be242665a72599748066661647f`
