# Run432 independent review — Astra

2026-09-27. Read-only script/result/raw-output review with pure-Python route statistics and sensitivity arithmetic. No framework import, NPU, service, or script mutation.

## Verdict

**Accept a narrowly conditional sampled eager token-identity result. Do not promote an exact `(token, top-k slot, expert)` contract, graph replay certificate, or all-43-layer row certificate.** Run432 makes useful progress on Run430's first unknown, but two oracle gaps prevent fully closing it.

Evidence is consistent: `result.json` reports three passing cases; stdout reports pass/3; stderr contains one CANN/HDK allocator32-padding warning, no recorded traceback. No separate process exit-code artifact was supplied; script main returns0 on recorded pass, which is consistent but not an independently recorded launcher exit status. This review does not re-certify service isolation from a `pgrep` snapshot.

## What the actual assertions establish

| Case | Geometry N/K/H | EP range | Valid / excluded | Observed max output error |
|---|---|---|---|---|
| small | 4/2/4096 | [0,32) | 4 / 4 | 0.001953125 |
| captured Target layer0 routes | 96/6/4096 | [0,32) | 85 / 491 | 0.002197265625 |
| same routes, nonzero range | 96/6/4096 | [96,128) | 81 / 495 | 0.001953125 |

The production route input used here is **96×6**, from Run403 rank0/cohort1/cycle64/Target entry0. Run430's proposed “actual96×K8” confused the verification block size8 with router top-k; the Run432 source-backed K6 is the correct geometry for this fixture. Verification rows per request and routed experts per row must remain separate fields.

For each case the script invokes Python `npu_moe_init_routing_v2` with E256, quant_mode1, explicit row_idx_type0 and a32-expert range. On the Run430 reviewed A2/CANN9.1 adapter path this is the V3 native dispatch. It checks:

- Exact local expert histogram and count sum.
- Every valid returned index is unique and lies in `[0, valid_routes)`; thus valid indices cover that prefix.
- Every excluded index equals -1 in these samples.
- At every valid returned index, the INT8 packed row's maximum-magnitude coordinate among the first N features identifies the original token t. This is a genuine check against routing output, independent of unpermute.
- A separate synthetic BF16 payload is placed at returned indices; output of native unpermute with `abs(indices)` and independently zero-masked probabilities is compared with token/slot-signature weighted sums. Other feature columns are required to remain exactly zero.

The96-row cases have38 and42 tokens with zero local routes, and21 and24 tokens with multiple local routes, respectively. They exercise repeated expert assignments across tokens, valid packed position0, excluded-slot aliasing through abs(-1), and both zero-origin and nonzero-origin expert ranges. This is more than a shape-only or counts-only test.

## Blocking gaps for a stronger certificate

### P1 — packed expert-segment association is not checked

For `(t,k,pos)`, the packed check only tests token t. It does not check that pos lies in the expert segment for `ids[t][k]`, using cumulative counts. Two routes from the same token to different experts can exchange returned positions and retain the same argmax identity. The later synthetic payload is populated using these same returned indices, so that step cannot independently detect this expert-slot mismatch. Histograms alone do not bind each position to an expert.

Minimum CPU-side oracle change: derive expert block boundaries from independently computed counts in ascending active-expert order; require `start[e] <= idx[t*K+k] < end[e]` for every valid route. Do not require an arbitrary stable tie order within an expert unless that order is separately specified. Preserve token check. This tests the complete `(token, expert)` association; fixture routes should be distinct per token, or duplicate expert slots need a separately stated equivalence convention.

### P1 — unpermute acceptance tolerance does not distinguish all slot errors

The signatures are `1+t/128+k/16` and weights `2^-(k+1)`, with a uniform tolerance0.04. A k4/k5 swap changes a two-route weighted sum by only0.0009765625 (pure-Python arithmetic checked here), well inside tolerance. Some missing late-k contributions are also below0.04. Thus low reported error and assertions do not certify every k-slot or exactly-zero masked output.

Minimum oracle change: use separate feature channels per k and powers-of-two exactly representable payloads/weights, making each route's expected nonzero channel an exact distinguishable integer; require exact equality where the chosen BF16 arithmetic is exactly representable. Include two nonzero routes and zero-local tokens; assert the latter output is exactly zero. Add CPU mutation checks for swapped k, omitted late-k, unmasked -1, wrong token, and wrong expert segment. Avoid an empirical tolerance that accepts these mutations. These CPU changes can establish oracle sensitivity before any next device run.

### Other scope limits

- `scales_finite_on_valid_prefix` is recorded as a boolean but **not asserted**. All three recorded values are true, but future false could coexist with status pass. No positive/nonzero scale assertion or dequantized payload comparison exists. The argmax test establishes a limited INT8 token marker, not complete quantized data/scale pairing or quantization accuracy.
- No full-range control was run, although Run430 proposed one. No claim about full-range behavior follows.
- The unpermute receives synthetic BF16 packed output rather than routing's INT8 output passed through real GMM. That separation is appropriate for an index test, but proves no GMM row preservation, expert weight identity, or numerical model correctness.
- Only two of eight possible32-expert ranges and one layer's route fixture are covered. No all43 layers, all-rank groups, other batch buckets, H values, routing branches or loaded template coverage.
- Explicit row_idx_type0 agrees with the reviewed intended default; the script bypasses actual dispatcher expert_map construction and communication. Range membership masking is an independent oracle, not evidence that the runtime expert_map was correct.
- Result stores package version but not CANN build, loaded shared-object/object SHA, tiling key, source fixture SHA, script SHA, device SKU, or assertion-enable state. Current file hashes below provide review-time identity; they do not prove the executed file had that hash. Ensure launch omits Python `-O`/PYTHONOPTIMIZE because the oracle uses `assert`.

## Permitted promotion

`MOE_EAGER_SAMPLED_TOKEN_GATHER_AND_MASK_OBSERVED`: for the three recorded geometries/ranges and fixture, local gather indices bijectively cover the packed prefix, point to rows retaining the source-token marker, excluded indices are-1, and abs-plus-zero-mask unpermute agrees with synthetic reference within the recorded errors.

Do **not** rename this `MOE_ROUTE_SLOT_CONTRACT_CLOSED`. After the two P1 oracle fixes pass mutation checks and an isolated rerun, a stronger sampled `(token,expert,slot)` contract can be considered. Exact loaded binary and test provenance remain part of that gate.

## Remaining graph / all43 prerequisites

1. Capture/replay the same native pair with graph-owned input/output references, two distinct sentinel generations and explicit replay entry identity; prove no stale routing/index buffers after input reuse. Eager success does not cover ACL graph lifetime or stream ordering.
2. Bind all43 actual Target layer branches, ordered groups/expert maps, runtime graph bucket, actual kwargs input storage, and selected-cycle metadata as specified by Run424.
3. Close remaining native query-row attention, rotary, GMM, A2A/fusedRS contracts for selected branches; this standalone single-device test exercises none of their distributed/native transitions.
4. Join request/position/target_logits_indices and Run420's authoritative external retention ledger. Neither acceptance nor external publication membership is tested here.

Recommended next step: strengthen the CPU oracle and its mutation tests first. A subsequent isolated rerun with those checks, then a two-generation graph replay sentinel, is more discriminating than another full-model run. No additional NPU execution occurred in this review.

## Review-time SHA256

- `scripts/loop079_moe_row_sentinel.py`: `fe2538509714636541cbf36d66eb7b3dab09a6e1ef161fda6c0257b8f62b2828`
- `evidence/20260927_loop079_identity/run432/result.json`: `56b1fdca7d24508c44e1b6223c255367680ca4102b78cf2a40d7e4e642911686`
- `evidence/20260927_loop079_identity/run432/stdout.txt`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `evidence/20260927_loop079_identity/run432/stderr.txt`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `evidence/20260927_loop078_bound/run403/capture/rank0_cohort1.json`: `334d7bdf8922d8d6790245681c0faddbcf899aeb1b1fd051e509d51d20455b62`

## V2 follow-up — supersedes the two P1 findings for tested mutations

Current script SHA256 `fe2538509714636541cbf36d66eb7b3dab09a6e1ef161fda6c0257b8f62b2828`. Parent preserved v1 as result_v1.json. V2 result reports3/3 pass, matching-expert-count-segment true, and max errors0.001953125 /0.0068359375 /0.0107421875 under new0.015 threshold. Source now checks each returned pos against the independently verified expert count prefix, before token argmax. This closes the previously omitted expert-segment association for these samples. Signature spacing changes from k/16 to4*k.

Executed stdlib-only `cpu_mutation_oracle.py` (exit0; result in cpu_mutation_oracle.json). It extracts the actual segment assignment/assert loop from current source AST, builds an independently sorted synthetic gather map, and swaps same-token/different-expert slots. All64 available pair mutations fail: small1, rank0 fixture33, nonzero fixture30. No same-token duplicate experts were present in tested valid slots. Because the inequalities reject different expert intervals independent of within-expert tie order, the check does not rely on stable ties. Full device indices are not saved, so this is an oracle sensitivity test, not replay of native raw index outputs.

A separate CPU BF16 round-to-nearest arithmetic check covers1440 pairwise k swaps (96 token signatures ×15 pairs). All exceed0.015; minimum error0.0579833984375. This resolves the earlier demonstrated late-k pair-swap blind spot. It is not exhaustive for arbitrary compound permutations, native accumulation strategies, or cancellation across many simultaneous corruptions. Separate k feature channels remain a stronger future exact oracle; no additional NPU run is needed merely to claim this pairwise sensitivity result.

Revised permitted promotion: `MOE_EAGER_SAMPLED_TOKEN_EXPERT_GATHER_AND_MASK_OBSERVED`, with k-pair mutation sensitivity. Valid route positions match token markers and independently counted expert segments; combined output passes the tightened sampled numeric criterion. Scale finiteness remains recorded rather than asserted, quantized scale accuracy is still untested, and binary provenance/all43/graph prerequisites remain as above. Run430 K8 references corrected to the actual96×6 fixture. Review performed no framework import, service or NPU action.
