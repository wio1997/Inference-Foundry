# Run414 — Astra independent communication Resource/Hardware audit

Date: 2026-09-27. Frozen contract: DeepSeek V4 Flash W4A8, 8 x 910B3, DP1 x TP8, DSpark7, 48 x 32K -> 1024, c12.

## Verdict

A numeric model-semantic network-byte floor, physical network latency floor, critical-join latency floor, and Product TPS ceiling remain UNKNOWN. Run391 establishes a current API tensor inventory. A useful conditional dense-transport cut calculation is possible, but its assumptions explicitly forbid several semantic-preserving changes that this product is allowed to make. It must not populate the compulsory network bytes field.

This review only read source and saved evidence, then wrote this document. It did not query or operate NPUs, execute profiling, alter borrowed source, touch the Run410/B service, or launch/stop processes.

## Evidence and provenance

Repository-relative evidence paths below are under /data/wio/Inference_Foundry. Source paths prefixed ASCEND are under /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend.

- evidence/20260927_loop077_bound/run391/findings.md: CANN 9.1 ABI and ordered source attribution for 265 Target operations.
- evidence/20260927_loop077_bound/run378/ledger.json, ordered_tasks: 265 ordered kind/count/dtype records, used directly for the calculations below.
- evidence/20260926_loop061_bound/environment.txt: saved topology at 2026-09-26 01:07:34 UTC. All off-diagonal pairs are labeled HCCS. The legend says "Connection traversing HCCS"; it does NOT establish 28 independent full-bandwidth direct links, lane counts, switch paths, simultaneous bisection capacity, or rank-to-physical-port routing.
- Latest Run246 rank0..7 ASCEND_PROFILER_OUTPUT/communication_matrix.json: all eight files are 39 bytes with empty p2p and collective objects. Rank0 and rank1 communication.json inspected: operation elapsed intervals exist but transit bytes/time are zero and the interval is assigned to Idle. These are missing transport observations, not proof of zero traffic or all-idle hardware.
- evidence/20260926_loop062_nongmm/findings.md: DSpark-unused MTP stash removal eliminated an AllGather and copies; no exposed Product gain established. Run267 shows collective duration includes rank-arrival waiting.
- evidence/20260927_loop078_bound/run400/dependency_ledger.json: source/Host/device mixed dependency skeleton, explicitly not a numeric finish-to-start DAG.
- evidence/20260926_loop074_refill/run339/findings.md and analysis.json: isolated attained HCCL service, not a physical capacity ceiling.

Current vllm-ascend git HEAD read during this review: 36589852a1eb8f5e842ad920f7c80ebdf1376ee9. Relevant source SHA256 read during review:
- ASCEND/attention/context_parallel/dsa_cp.py: 27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e
- ASCEND/ops/fused_moe/prepare_finalize.py: c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd
- ASCEND/ops/vocab_parallel_embedding.py: a4a8517abc3a4f8494f27cc4cc3a876264ef8fea8cfc6de9abdbaf75d42c3b24
- ASCEND/models/deepseek_v4.py: 11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247

These are current inspected sources, not a claim that every loaded native binary or old Run246 local modification has been independently matched.

## 1. Current API inventory

Run391's source/ABI-consistent totals are 115,107,840 input B and 145,342,464 output B per rank-cycle. Their sum is not network traffic. The reported count-times-dtype sum 25,651,200 B mixes API conventions.

For p=8 and C=reported count times dtype bytes:
- AllGather: input C, output 8C.
- ReduceScatter: input 8C, output C.
- AllToAll: input/output 8C, because count is per peer.

Current sequence:
- ordinal 0: embedding BF16 RS, C=98,304.
- For each of 43 layers: hidden BF16 AG C=98,304; DSA head/token A2A C=98,304 per peer; wo_b BF16 RS C=98,304; MoE hidden BF16 AG C=98,304; router FP32 AG C=12,288; MoE BF16 RS C=98,304.
- Tail: MTP residual AG C=393,216; four hidden/DSpark-aux AG each C=98,304; logits AG C=3,102,720.

This is RS87 + AG135 + A2A43 = 265. It describes padded96-row current Target, not a useful-row or parked-slot-eliminated algorithm.

## 2. Conditional dense-transport cut calculation

Define a logical rank cut S with k ranks versus the other 8-k ranks. Count bytes crossing the cut in each direction once, independent of further hops inside either side. Assume ALL of:
1. Fixed current tensor ownership and all current API output materializations.
2. Each operation transports fresh raw values; no cross-operation reuse, reconstruction from other tensors, weight migration, alternate placement, zero elision, lossless compression, lossy quantization, or pruning of dead/parked/rejected rows.
3. Payload words are treated as opaque necessary words. ReduceScatter permits within-side aggregation into the same-size reduction words. This is an ordinary dense communication model, not an entropy theorem for the actual model values.
4. No required source values already reside on the opposite side before this operation. Future consumer placement is unchanged.
5. Sum only bytes across operations. No sum of measured operation latencies is used.

Then a relaxed directional cut floor for each primitive is:

| Primitive | S -> complement | complement -> S | Two-way total |
|---|---:|---:|---:|
| AG(C) | k C | (8-k) C | 8 C |
| RS(C) | (8-k) C | k C | 8 C |
| A2A(C per peer) | k(8-k) C | k(8-k) C | 2k(8-k) C |

For RS, the first direction carries one aggregated remote-side contribution for each destination chunk outside S. These formulas permit ideal aggregation/broadcast within each side and omit protocol/header/retransmission overhead. They are not predictions of a native HCCL algorithm.

With H=98,304 and R=12,288, per layer:
B_layer_cut(k) = 32H + 8R + 2k(8-k)H.
At k=4 this is 6,389,760 B, of which 3,194,880 B travels in each direction.

Whole Target includes the embedding and tails:
B_cycle_cut(k) = 43 B_layer_cut(k) + 8H + 8(393,216 + 4H + 3,102,720).

| Cut size k | S -> complement B | complement -> S B | Total B/cycle |
|---|---:|---:|---:|
| 1 | 102,328,320 | 128,243,712 | 230,572,032 |
| 2 | 127,782,912 | 145,059,840 | 272,842,752 |
| 3 | 144,783,360 | 153,421,824 | 298,205,184 |
| 4 | 153,329,664 | 153,329,664 | 306,659,328 |

Reproducible pseudocode:
```python
tasks = json.load(open("evidence/20260927_loop077_bound/run378/ledger.json"))["ordered_tasks"]
for k in (1, 2, 3, 4):
    outgoing = incoming = 0
    for op in tasks:
        C = op["reported_chunk_bytes"]
        if op["kind"] == "hcom_allGather":
            x, y = k*C, (8-k)*C
        elif op["kind"] == "hcom_reduceScatter":
            x, y = (8-k)*C, k*C
        elif op["kind"] == "hcom_alltoall":
            x = y = k*(8-k)*C
        else:
            raise ValueError(op["kind"])
        outgoing += x
        incoming += y
    print(k, outgoing, incoming, outgoing + incoming)
```

Confidence: high for arithmetic under the stated transport model; LOW applicability as a semantic lower bound. The following section supplies concrete violations of its assumptions. Do not describe 306.659 MB as compulsory model network traffic.

## 3. Model information versus placement/materialization

| Current operation | Information consumers need | What current bytes do not prove |
|---|---|---|
| Embedding RS | Correct embedding for each live token at its next compute owner | Only the token's vocabulary owner contributes nonzero values. ASCEND/ops/vocab_parallel_embedding.py:229-248 explicitly masks other ranks to zero. Owner-directed transfer, reused embeddings, or different embedding placement can replace dense reduction. Nonlocal owner count is needed for a fixed-placement conditional byte ledger. |
| DSA hidden AG | Correct Q/KV/Compressor inputs at their compute/state owners | ASCEND/attention/context_parallel/dsa_cp.py:1435-1444 gathers hidden globally, while local Q begins from hidden_states_local at :1449. Full96 hidden replication is a placement decision; owner-only producer work and moving projections change communication. Local producer parity evidence exists elsewhere, but full persistent replacement is not proved here. |
| DSA A2A | Attention results must reach the owner of the corresponding output projection contributions | Current :1645-1675 exchanges token-owned full heads into head-owned all-token tensors. The semantic requirement is the resulting wo_a/wo_b contribution, not this exact layout exchange. Moving groupwise wo_a before transfer with suitable weight placement can change representation size (4096 group input to1024 output); numerical/quantization parity and additional weight/HBM costs require validation. Source already has a conditional skip_all_to_all/full-weight path, not evidence it is a Product win. |
| wo_b RS | Contributions from tensor-sharded projection weights must be combined | A join is necessary under this fixed sharding when remote contributions matter; raw96x4096 per-rank partial buffers and this collective algorithm are not invariant. Output ownership, fusion, moved compute/weights and alternative exact encodings trade other resources for link traffic. Source :1343-1391 and ASCEND/ops/linear_op.py callsite mapped by Run391. |
| MoE hidden AG | Each selected expert owner needs sufficient input and routing information; token owner needs its weighted result | Only expert destinations actually selected by a token need its input. Broadcast to every rank is not semantic. Per-token expert destination IDs, request row ownership and active masks are necessary to count fixed-placement cross-cut dispatch. Aggregate expert counts alone cannot determine it. Quantized activation plus exact scales may be sufficient at an appropriate verified boundary; BF16 payload size is not automatically compulsory. |
| Router FP32 AG | Correct top-k experts and routing weights | ASCEND/models/deepseek_v4.py:382-385,:474-480 computes router from hidden and replicated gate weights. ASCEND/ops/fused_moe/prepare_finalize.py:386-387 gathers both. Remote logits are derivable once hidden is present. Recompute or transmitting selected IDs/weights changes bytes. Exact numerical routing parity is required, but the full logits AG is not an independent information floor. |
| MoE RS | Weighted sum of selected expert contributions at the next token owner | Same fixed-placement join obligation, but inactive expert/token contributions need not be transmitted as dense zeros. Within-rank aggregation, destination-directed return and expert placement change traffic. Source prepare_finalize.py:503-524. |
| MTP residual tail AG | No DSpark consumer established for this stash | Existing Run257/263 consumer audit and Run260/261 removal establish that this current collective can disappear. The current 265-operation census therefore cannot be a compulsory count. No E2E gain follows from the removal. |
| Hidden/three auxiliary tails | Target head and DSpark need the appropriate hidden/aux values for retained context/proposals | Full96 rows on all8 ranks, separate gathers and current consumer placement are not proven minimal. DSpark main projection consumes three target auxiliary layers; moving that projection changes the representation/ownership tradeoff. Dead suffix and parked-row consumers must be closed first. |
| Logits AG | This fixed Runtime's greedy verifier needs target argmax IDs, with exact tie/nonfinite behavior | runtime/fixed_acceptance.py:27-35 immediately takes argmax and calls greedy_accept. Distributed local-max/value-ID selection can compute the same greedy result without materializing all129280 logits on every rank, subject to exact dtype/tie/NaN semantics. Full logits may still be required in other sampling contracts; this statement is limited to this source path. |

For finite logits and canonical ties, a constructed distributed argmax protocol can exchange one local max plus global token ID per row and combine candidates. For illustration, an 8-byte candidate record gives 96x8=768 B of local candidate records versus 3,102,720 B of local full logits. This is a possible alternative communication representation, NOT a lower bound, implemented optimization, measured saving, or proof of exact parity. Keep head compute and any new collective/copy costs in its eventual assessment.

Freezing DP1TP8 does not by itself freeze every activation/weight/consumer placement or demand all intermediate tensors be replicated on all8 ranks. Consequently none of the raw-API dense cut totals is a model-semantic invariant. Exact compression can exploit zeros/duplicates but its obtainable ratio is unknown; arbitrary lossy quantization is not authorized by W4A8 alone.

## 4. Critical joins and hardware time

For fixed current placement, source gives real partial orders:
- Required DSA result exchange -> wo_a/wo_b contributions -> their reduction result -> following layer state.
- Routed expert evaluation -> required contribution combination -> subsequent nonlinear layer computation.
- Target argmax/acceptance -> retained state/DSpark seed -> seven Markov feedback steps -> next Target candidates.
- Relevant peer-produced data must arrive before the consumer reads them.

These show dependent joins, not a mandatory count of 265 global barriers. Streaming/tiled producers and consumers may overlap different tensor portions. Host enqueue order, task launch, rank arrival, transfer completion and downstream consumer completion are distinct. Run267's arrival-heavy first RS and Run400's mixed edge semantics specifically preclude adding reported task duration or solving its raw edge list as a weighted serial chain.

For a proven physical cut with directional throughput CEILINGS U_out and U_in, a valid conditional traffic ledger would imply:
T_network >= max(B_out/U_out, B_in/U_in).
If the physical resource shares a single aggregate bidirectional capacity U_sum, also T_network >= (B_out+B_in)/U_sum.
Use only the applicable resource model; do not add these lower bounds. A resource-constrained DAG may strengthen them after node work, true joins and overlap are measured.

No such physical capacity ceiling has been established here. HCCL Test attained bandwidth is a achieved performance sample, generally a LOWER sample of attainable throughput, not an upper cap on all valid algorithms. Dividing compulsory bytes by that sample does not prove a hardware latency floor. The HCCS reachability matrix does not close this gap. Whole Product also needs Draft, prefill/seed, acceptance, publication and legal arrivals; this inventory contains Target only.

Confidence: high that joins exist under current placement; UNKNOWN numeric critical-join latency or complete algorithm-independent network requirement.

## 5. Next minimal experiment

After the active run has completed and its sources are restored, first do a ONE-collective instrumentation capability gate on the actual Target DSA A2A at a selected layer/cycle. Capture all8 native communication peer/source/destination, per-transfer bytes, transport/link identity and local completion records, preserving the API ordinal. Require matching send/receive records and nonempty transport observations. Existing Run246 exports are empty for this purpose, so repeating generic Level1 collection without first proving transport export is not useful.

This gate should answer whether the observed A2A really transfers 7x98,304=688,128 nonself payload B per rank at endpoints, how that maps to transport paths, and which exported records represent actual movement versus waits. Endpoint payload is not physical multi-hop bytes. If the gate cannot expose native movement, record instrumentation UNSUPPORTED; use source/ABI conditional ledgers without inventing physical measurements.

Do not time-sum the trace or infer a capacity ceiling from its observed rate. The immediate value is resolving API-to-transport ambiguity and choosing a real physical cut. Only then extend to the 265-operation chain and obtain separately justified hardware capacity ceilings/mixed-resource calibration. If semantic MoE dispatch bytes are the next numerator priority, pair the already captured expert IDs with propagated row-owner identities; Run405 says that row mapping is not yet closed.

## Appendix — retained-row relaxation requested during the earlier audit

This appendix clarifies the earlier Run375 expert-union interval, not a new network bound.

runtime/greedy_accept.py:42-55 sets a=number of leading accepted drafts and count=a+1. When a<7, rows0..a-1 certify the accepted drafts and row a supplies recovery/fallback. When a=7, rows0..6 certify drafts and row7 supplies bonus. Thus raw count is the number of leading Target prediction rows needed by this ordinary greedy verification, not merely the number of accepted drafts. Fallback is already included: do not add one again.

For each request i let r_i be remaining externally requested output and q_i=min(raw_count_i,max(0,r_i)), with q_i=0 for an inactive/parked slot. Under a hypothetical truncation-aware, causal-prefix execution, its retained output requires the first q_i prediction rows. If overshoot clips before recovery/bonus, that discarded recovery/bonus row is not required for the retained external prefix. However current Runtime may still compute/update it, and later persistent state or Draft consumers may require more than the retained-output subset. Proving complete state liveness is separate. Initial/handoff outputs already published must be subtracted from r_i; do not equate Runtime count with whole-client output cardinality without that ledger.

For each layer, existing full96 route counts c_e are upper capacities for a retained U=sum(q_i) row subset. Assume each row selects six DISTINCT experts, all43 layers preserve the relevant causal row semantics, ordinary routed expert evaluation, unchanged routing, and no previous result reuse. Let s_e=min(c_e,U). Sort s_e descending. The smallest k with sum(s_e[:k])>=6U is a lower relaxation on the selected expert union. Its upper enclosure is min(number_of_positive(c_e),6U). Sum layer bounds and multiply12,582,912 packed B/expert. This is an enclosure of a selected WEIGHT STORAGE SET, not necessary HBM traffic or a compute schedule. Token co-occurrence, per-request retained masks and layer-to-layer row correlations are discarded, so the endpoints need not be attainable.

Reproducible pseudocode:
```python
U = sum(min(raw_count[i], max(0, remaining[i])) if active[i] else 0 for i in slots)
lo = hi = 0
for layer in layers43:
    counts = concatenate(all8_rank_route_counts[layer])  # 256 experts
    assert sum(counts) == 96*6
    if U == 0:
        continue
    caps = sorted((min(c, U) for c in counts), reverse=True)
    k = first_k_with_prefix_sum_at_least(caps, 6*U)
    lo += k
    hi += min(sum(c > 0 for c in counts), 6*U)
packed_set_interval = (lo*12582912, hi*12582912)
```

Run403/405 later observed per-token IDs, but Run405 explicitly leaves propagated row identity conditional. That newer limitation also applies to any exact retained-mask union based on those IDs. Neither the older enclosure's upper endpoint nor the newer observed union is "necessary weight traffic."
