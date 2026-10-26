# Run520 — Astra independent compulsory-work census plan

2026-09-27. Frozen contract: DeepSeek V4 Flash W4A8, eight Ascend 910B3, DP1×TP8, DSpark7, warmed 48×32K→1024 at c12. Formal Current remains **571.681 tok/s**. Source/header review and CPU integer arithmetic only; no service, NPU query/workload, TaskCtl/model/source edit or endpoint promotion.

**Recommendation:** replace the one-group candidate with a source-derived, per-fresh-required-Target-row **all43-layer wo_a census**. This is the smallest extension with a materially larger numerator while retaining the same BF16 capacity class and avoiding routed-expert identity as a prerequisite. Then add the route-cardinality-invariant MoE census under its own W4A8 arithmetic/capacity contract. The crucial next proof is the multiplicity of fresh required rows, not another current-route counter census.

This produces a useful **conditional conventional-evaluation bound**, not a universal arithmetic-complexity lower bound for the unrestricted architecture search. No source census alone can honestly promise a numerical strict Product endpoint: formal freshness/retention and genuine C-plus/B remain independent gaps.

## 1. What can be derived now

The pinned checkpoint headers contain exactly43 Target wo_a tensors, layers0–42, each BF16 [8192,4096]. Config has hidden4096, 64 heads×512, 8 output groups, output rank1024,43 layers. Source constructs the grouped projection and calls it in every decoder layer. Thus each semantic row has8 independent4096→1024 group projections; TP8 distributes groups but does not introduce another factor8.

| Subset | Semantic shapes and count | Conventional work per fresh required Target row |
|---|---|---:|
| Existing witness | One [1,4096]×[4096,1024] group in layer42 | 8,388,608 BF16 operations |
| Recommended BF16 census | Same group ×8 groups ×43 layers | **2,885,681,152 BF16 operations** |
| Routed MoE next tier |43 layers ×6 expert incidences; gate/up jointly [1,4096]×[4096,4096], down [1,2048]×[2048,4096] | **12,985,565,184 GEMM-equivalent operations** |
| Shared expert, optional later | One2048-wide expert per layer, same two GEMMs |2,164,260,864 GEMM-equivalent operations |

MAC=2 is a work-unit convention. The MoE count is43×6×50,331,648, independent of which six experts were chosen and how ranks own them. Actual distinct expert IDs affect reuse/traffic and placement, not this ordinary incidence arithmetic. The shared-expert row is configuration-derived and must be bound to the actual shared/fused path before admission; it is deliberately outside the first two certificates.

The all43 wo_a work is **344×** the one-group witness for one fresh row. If a separate proof established F=49,152 fresh required row evaluations in the timed interval, this subset alone would contain141,836,999,983,104 operations, or16,908,288× the original one-group numerator. This is a sensitivity example, **not an asserted F or new bound**. The same hypothetical MoE numerator is638,266,499,923,968 conventional operations.

Checkpoint header inspection is stronger than a profiler-shape guess but is not a hash of weight payloads, proof of nonzero/algebraically independent weights, or historical loaded-image certification. The manifest records each file/header hash, tensor offset, dtype and shape. An actual loaded-branch/evaluation-class join remains necessary.

Source anchors: deepseek_v4.py attention construction around729–800; unconditional MoE construction947; decoder forward985–1000; ordinary BF16 dsa_cp.py1374–1391. Run370 independently corroborates43 observed transpose-BMM tasks of M96,K4096,N1024 per rank/cycle:34.628173824 GF/rank/cycle. Multiplying by8 ranks and dividing by96 semantic rows gives2.885681152 GF/row, matching the source derivation. This cross-check is about dimensions, not observed-work necessity.

## 2. Declare the legal class before calling this W-minus

A required semantic function is not a lower bound on arithmetic to evaluate it. The model defines grouped projection and MoE outputs feeding later nonlinear/residual paths. It does not prove that every mathematically equivalent implementation must execute dense2MNK.

For the recommended certificate define a class in which each counted fresh semantic wo_a group is evaluated using the ordinary dense BF16 contraction with the stated accumulation/rounding semantics. Require that its value is needed by the admitted output and cannot already be supplied by an allowed cached intermediate. State whether factorization, decision-only output computation, exact sparsity shortcuts, input-specific simplification and alternate intermediate representations are excluded. Excluding output memoization alone is insufficient.

The routed-MoE tier additionally requires the six selected expert contributions to be evaluated conventionally, with the same quantization/dequantization and nonlinear semantics. Its50,331,648 unit count must **not** be divided by a BF16 peak merely because activations ultimately become BF16: packed W4A8 kernels may execute integer products, wider intermediates and scale operations. Either certify a matching conservative service envelope for these work units across all permitted engines, or leave this tier as a conventional arithmetic census only.

The product mission allows architecture changes. Consequently these restrictions are explicit conditional analysis classes, not silently added product requirements. There is no positive universal FLOP/HBM numerator proved here for all equivalent legal architectures and the finite repeated dataset. Current source execution, successful outputs and routing counters do not establish one.

## 3. The missing multiplier is F, with per-layer/group granularity

Use the exact full formal interval I and N=49,152 externally admitted tokens. Define semantic evaluation keys by request/context-state identity, position, layer, group, model/weight identity and relevant KV/input generation. Define F[l,g] as the number of distinct required fresh evaluations of that group attributable inside I, after permitted reuse and pre-window state are accounted for. Then:

W_wo_a_minus = 8,388,608 × sum over l=0..42,g=0..7 of F[l,g].

Only if all344 entries have a proved common minimum F may this be simplified to2,885,681,152×F. Equal output token IDs do not imply equal contexts or reusable projections; different request IDs do not prove distinct contexts. Batching the same weight over many rows counts each conventional row contraction, but weight reads can be shared.

Do not set F from current1212-ish cycles×96, assume96 active rows, divide by mean acceptance, or use the512-cycle current-cohort cardinality relaxation. These are implementation/trajectory objects. Do not set F=N just because each completion has1024 tokens: cached prefix/logits, pre-window evaluation, cross-request duplicate contexts, allowed warm-pass memoization and layer-intermediate reuse must first be resolved.

A useful bounded source lemma extends Run497: for leading accepted count A, **every retained sampled position j=0..A equals Target argmax P[j]** (j<A copied a matched draft; j=A is recovery or bonus). This enables a semantic retained-prefix census without assigning compulsory work to rejected proposal suffixes. Existing Run497 executed the first-position claim only; the extension needs its own pinned-source CPU check across128 patterns, counts/masking/clipping, and cannot infer formal freshness.

Proposed narrow next deliverable: one table per admitted request containing selected Target evaluation keys, Runtime masked prefix q/R, exact Scheduler pre-append G and admitted prefix, external ordinal/representation relation, and pre-window/reused/fresh status; aggregate only proved fresh keys. In the first acquisition, a sparse contiguous retained prefix can be enough—do not demand all43 native per-layer route pointers merely to establish the BF16 source-class numerator. If source layer semantics and valid transformation contracts do not establish the selected row chain, retain its missing membership gate explicitly.

Run517 shows accepted Run99 artifacts cannot reconstruct even one complete fresh formal witness. An offline planner cannot repair that. Reuse Run517’s exact owner/observation-phase specification if later acquisition is authorized; preserve full formal48 timing even if sparse witness data are retained. For a class requiring literal current native all43 provenance, the existing FlashComm/DSA/row-layout proof gaps remain; do not hide them with the source-level count.

## 4. Target, Draft, Prefill and KV remain separate

**Target.** The recommended subset counts only required retained causal rows. Rejected suffixes, padding, inactive slots and overshoot stay in the current-work ledger. Row ancestors genuinely needed for a retained output may contribute, but count by semantic key to avoid duplicate charge. First-token evaluations performed during residual prefill can belong to the same semantic ledger if fresh in I; do not count them again as decode.

**Draft/DSpark7.** Frozen DSpark7 does not imply seven copies of the43-layer Target. Config points at target layers40,41,42; Run372 sees three GMM pairs plus28.805300224 GF/rank/cycle in a partial dense census, with asynchronous tail tasks extending beyond the Host scope. Whether and how many draft evaluations are mandatory follows from the admitted DSpark execution semantics, required proposal positions and opportunities for reuse, not observed cycles. Draft route counts, fresh shared intermediate ownership and all proposal/verification work remain distinct. A Target-only relaxation may omit Draft safely; this weakens it.

**Prefill.** Keep P_fresh(request,position,layer) symbolic until the actual prefix-cache lookup/retained-block state at admission and required residual/chunk rows are known. Run242/239 residual shapes are a separate diagnostic and cannot supply formal counts. The parent’s Run99 server-log clue—cumulative cache-hit changes and ~99.6 prompt tokens/s windows—is not an exact timed residual-prefill numerator; window attribution, tokenizer rows, eviction and counter denominators remain unknown. Warm-cache does not mean zero Prefill, and32K×48 is not compulsory uncached work. Counting only certified Target decode rows is conservative; completing Prefill is valuable later but not prerequisite to the first larger subset.

**KV/attention.** Config identifies43 layers with ratios0×2,4×21,128×20, SWA window128, head_dim512, index_topk512 and index_head_dim128. These specify semantic structures, not compulsory per-query HBM bytes. Actual valid length, selected compressed positions, shared cache blocks, speculative committed versus discarded writes, storage replication, dtype/packing and warm resident state matter. Avoid charging128×512×2 bytes as an unavoidable SWA read per query: keys/values overlap across heads, batch rows and adjacent queries; fused attention may retain or recompute them. Similarly, c4/c128 state footprints in Run242 are not transfer lower bounds.

## 5. Traffic: only a finite-storage/recomputation argument is meaningful

The43 BF16 wo_a tensors have a verified **2,885,681,152-byte aggregate storage footprint**, or360,710,144 bytes/rank under balanced one-group-per-rank placement. This numerical equality with operations/row is accidental (BF16 bytes and MAC=2). It is not2.885GB HBM traffic per row/cycle.

For a restricted representation requiring these distinct stored coefficients to be accessed from HBM and disallowing equivalent coefficient generation/compression/recomputation, a first interval-level read relaxation is:

Q_HBM_minus >= max(0, required distinct coefficient bytes - initially available fast-memory bytes).

The fast-memory allowance must cover all participating L2/L1/UB/register/other allowed stores and initial residency, with legal ownership/transfer accounted for. Cross-device links and host supply cannot be silently treated as free or silently banned. Warmup may leave many coefficients resident; assume worst permissible initial residency for a lower bound. A header-storage footprint alone does not prove all bytes are semantically needed.

This once-per-interval relaxation may still be weak. To obtain repeated compulsory transfers, use ordered semantic phases and a proved finite-storage/pebble-game or reuse-distance argument allowing the permitted batch reordering and recomputation. At each proved cut, charge only data that cannot survive in fast storage and must be obtained again; do not sum per-cycle footprints. Count compute-for-recomputation and traffic alternatives jointly when the legal class allows that tradeoff.

Run368’s8.95–9.58GB active packed weights/rank/cycle and ~1.071 read-to-footprint ratio establish a current route/counter relation only. Run380’s7.80–9.01GB range and varying19.18–38.96 GF per Runtime-useful token likewise cannot become compulsory per-token traffic. Run256’s measured compressor reads include replication/reloads; its three projection shapes are current implementation arithmetic, not three universally necessary HBM passes. None should replace the finite-storage proof.

**Priority:** complete the BF16 work/multiplicity ledger first. A rigorous traffic bound requires storage and representation assumptions absent from the present evidence; recording those variables is useful, but another measured-byte census does not close them.

## 6. Bounded execution plan, confidence and stopping rule

1. **Completed in this review:** read the specified evidence, config and source; parse only safetensors headers; verify43 Target wo_a BF16 [8192,4096] tensors; derive integer arithmetic and storage totals. High confidence in these static dimensions and arithmetic. Medium confidence in transfer to any selected live branch until its load/branch certificate is joined.
2. **Next offline artifact:** a small class-labelled semantic census containing344 wo_a entries and258 routed-MoE incidences per fresh row; bind source/layout contracts and explicit exclusions. Add the full retained-prefix CPU lemma and negative controls for stale generation, wrong prefix, pre-window cached value and duplicated semantic keys. High expected information value; no profiler/service required.
3. **Missing evidence specification:** derive the minimum formal48 fresh-key/retained-output record from Run517. Keep F[l,g] unknown where absent. A new measurement, if later authorized, should acquire this scalar/semantic census without per-layer barriers or broad all-rank route profiling.
4. **Capacity remains parallel documentary work:** same BF16 ordinary work units, all admitted engines, actual board/bin, maximum clock/issue including tolerance/boost, cumulative service C-plus T+B. Do not use measured sustained capacity as C-plus. For the selected subset use T >= max(0,W-minus-B)/C-plus; its larger numerator may reduce relative boundary sensitivity, but no B is invented.
5. **Stop this census branch** once dimensions/class/multiplicity gaps are explicit. Further current-route or byte counters do not cure freshness or universal-capacity gaps. Continue separately with the highest-value missing proof rather than expanding to every tiny operator.

A larger W-minus strengthens the resource relaxation proportionally **only under the same valid class, F and matching capacity certificate**. It does not prove that the real achievable performance is close to the resulting ceiling. Other-resource lower bounds combine by valid overlap/dependency constraints; distinct engine peaks or node times must not be casually summed. V3.24/V3.25 correctly retain null endpoints and Current571.681; this review recommends no change to that status.

## Provenance

Machine-readable inputs and header census: `astra_compulsory_census_inputs.json`, SHA256 **6b0ad89b5fdd96bfad61b13c723c7baa3f4d6b204d50b9855c9d28cc729dc08a**. It includes hashes of every cited primary evidence/config/source file, all43 safetensors length/header pairs, exact Target tensor shapes/offsets and arithmetic. Header hashes certify inspected metadata only; weight payload bytes were not read. This audit used the current pinned source as a model-semantic reference, not as proof of the exact historical Run99 loaded object.

The full evidence/config/source hashes are appended below.


- `AGENTS.md`: `cd6e1e562d9a0f0a4ae5efbd60718dc85e0fcc3dc33fd4e555afa510d9375ec0`.

- `MISSION.md`: `fc2dc51304e85f69c4c69dcc20001225d87d006aec6a26ee0dfe0536227a12e8`.

- `evidence/20260926_loop060_resource/run242/inventory.json`: `0f897125ff729c17e0160b3a9a517b4a676059e2642ab23c08784de14f9c900e`.

- `evidence/20260926_loop062_nongmm/run256/census.json`: `d13a4cac23ef5a4665dd0f9e69ec45066df211af15bdf3cea7735a70b7c57c64`.

- `evidence/20260927_loop076_bound/run368/analysis.json`: `8abfd0d085c63377466a37ba2a47a18bb03a9e8862c8903b339068df6c28c931`.

- `evidence/20260927_loop076_bound/run370/analysis.json`: `6185c78dc5a13c1076a4bb164c6e120fea999f36130ba79ed109981b0b44d7fc`.

- `evidence/20260927_loop076_bound/run372/analysis.json`: `a3a447f58fe6e62972f345d62b56839cec181cba01f38d7f82659e9d8c6ec7be`.

- `evidence/20260927_loop077_bound/run375/bench.json`: `2601791adfb8b65d68d31970e3efa0fd0c5e1bfcf1bbcb66a4ce55d81e5e3d34`.

- `evidence/20260927_loop077_bound/run380/analysis.json`: `46c45b1053a6f9489fd2531c13321a754ddbfb3f463b4817b7f5f4976010d668`.

- `evidence/20260927_loop079_identity/run495/resource_first_position_design.md`: `097c8290c2c26e80e50994d5fe505dba9ae92728e44ad122cd0d9fb2147737c1`.

- `evidence/20260927_loop079_identity/run497/first_position_cpu.json`: `2c78373b7dbba68ad839cf9fe8c2575db14fb50ccd676b98d36fb616dd10e21c`.

- `evidence/20260927_loop079_identity/run509/astra_resource_certificate_review.md`: `570616b1fc6405bfa031ef3ae081c3ab98fe74b5d45ad68efd0faf04c8a1812e`.

- `evidence/20260927_loop079_identity/run514/bound_calibration_v3_24.json`: `71956b009bb160cf562d7135f9a963262e2088ab30e474596869f6bcaa3f985a`.

- `evidence/20260927_loop079_identity/run517/astra_formal_witness_inventory_review.md`: `e3b233ae7c185fdef1671755f8a810a3d6b4ea272c6fdeee36f2d68065d8177b`.

- `evidence/20260927_loop079_identity/run518/bound_calibration_v3_25.json`: `162c95002f8d021786c4fcf0a90129adf57c00b5218fe7cf4968b816fdbdb378`.

- `/data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json`: `6c6cdc4a47e00e41137081429b95bd5c1cf91ba62e87309d1398431de7f4a8e1`.

- `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py`: `11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247`.

- `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py`: `27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e`.
