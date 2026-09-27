# Run423 — Astra minimal strict Resource/Hardware ceiling review

Date: 2026-09-27. Scope: frozen DeepSeek V4 Flash W4A8, 8 x 910B3, DP1 x TP8, DSpark7, 48 x 32K -> 1024, concurrency12.

## Decision

A complete work census, mixed-occupancy model, or complete scheduling DAG is NOT necessary to prove some finite relaxed Product TPS upper bound. One positive compulsory work subset and a matching genuine capacity ceiling suffice. Earlier wording that required every missing work family and every dependency to be closed before ANY finite bound was too strong.

However, this review did not find a fully certified pair in the existing evidence. Keep the strict model-semantic Product ceiling null. Publish conditional candidates separately. This is an evidence limitation, not a theorem that a finite bound is impossible.

No service, NPU, patch, or borrowed source was touched. This review read saved source/evidence and public official documents by Web Search/direct HTTP; only this Markdown file was written.

## 1. Minimal theorem; omissions and overlap do not invalidate it

Let N=49,152 be external tokens for the frozen complete benchmark. Let W^- > 0 be work that EVERY allowed correct implementation must perform during the same measured wall interval. Let C^+ be an upper bound on the aggregate rate at which ALL allowed execution engines can perform that very work, in matching units.

Then:

T_product >= W^- / C^+
TPS_product <= N C^+ / W^-.

If W^- = N_new w for N_new newly evaluated required prediction positions, the corresponding bound is N C^+/(N_new w).

Omitting Draft, attention, metadata, HBM, communication, prefill, idle, and output drain makes this bound weaker, not false. Assuming perfect overlap and full theoretical capacity also makes it weaker. No measured serial sum or overlap estimate is needed for this first relaxation.

Requirements that cannot be omitted:
- compulsory means a lower amount for the stated implementation class, not the amount seen in one current execution;
- fresh work must fall inside the Product measurement interval, excluding work already done in warmup/prefix/output caches;
- external N and counted work must share the measurement boundary;
- C^+ must bound all ways of performing the subset that the claimed class permits;
- capacity units must match actual arithmetic/bytes, including MAC-versus-operation convention, precision and sparsity.

A conservative universal C^+ may intentionally be much larger than attained performance. That is valid if supported. Multiplying an unproven estimate by an arbitrary safety factor does not turn it into a certificate.

## 2. Actual hardware identity: established facts and missing binding

Saved evidence, without a new device query:
- evidence/20260927_loop079_identity/run421/preflight_npu.txt explicitly reports eight devices named 910B3, each with65,536 MB HBM. Bus IDs are C1:00.0, C2:00.0, 81:00.0, 82:00.0, 01:00.0, 02:00.0, 41:00.0, 42:00.0.
- Run246 profile device info reports hostname S900K3-49,20 AI Cube cores,40 AI Vector cores, observed AIC/AIV frequency1800 MHz. Example:
  evidence/20260926_loop060_resource/run246/profile/rank0_160893_20260925171535454_ascend_pt/PROF_000005_20260925171535504_00160893BIRQQPCN/device_0/info.json.0
  The same fields were observed throughout the saved all-rank info files.
- evidence/20260926_loop061_bound/environment.txt gives HCCS reachability among all8 devices, not lane/cut capacity.

The20-core count is platform evidence. Observed1800 MHz is not proof of a maximum permitted frequency, and these fields do not specify the maximum issue/completion rate of every relevant arithmetic engine. S900K3-49 plus 910B3 does not by itself identify an Atlas800T A2 purchase SKU or bind it to a white-paper rate option.

## 3. New primary vendor source, and why it is not yet the matching certificate

Official source: Huawei, Atlas800T A2 Training Server Technical White Paper, revision16,2026-05-12.
Landing page:
https://e.huawei.com/cn/documents/products/computing/b0b253c4d5da4e05ba80a39e3dc65fd2

Public attachment URL was read from the landing page's attachmentUrl field and retrieved by direct HTTP:
https://e.huawei.com/marketingcloud/pep/asset/20000001/Material/b0b253c4d5da4e05ba80a39e3dc65fd2/M3T1A669N1268492946326892689/Atlas%20800T%20A2%20%E8%AE%AD%E7%BB%83%E6%9C%8D%E5%8A%A1%E5%99%A8%20%E6%8A%80%E6%9C%AF%E7%99%BD%E7%9A%AE%E4%B9%A6%2016.pdf

Retrieved PDF:3,570,178 bytes; SHA25647fa3144e1dda9815a77451edf5f9babc6d82e589719b5e10edf921242bdd3fe.

Printed pp36-37 list optional313/376 TFLOPS FP16 and82/99 TFLOPS FP32 per processor,64 GB memory with1600 GB/s design bandwidth, and whole-system2.504/3.008 PFLOPS FP16. They describe eight-way full mesh and seven HCCS links per processor with maximum theoretical392 GB/s. The footnote identifies theoretical design specifications and says measurement may differ, with error range/details available from technical support.

This source is materially stronger than reseller tables, community blogs, or benchmark peaks. Nevertheless:
1. It does not map910B3 to either compute option, or establish that this OEM host obeys that exact configuration's maximum.
2. The inspected revision does not provide a matching W4A8 integer dense-operation ceiling. FP16 FLOPS cannot be directly paired with W4A8 standard2MNK.
3. BF16 equivalence, rounding/accumulation semantics, dense-versus-sparse treatment and whether the advertised rate covers all relevant simultaneous engines must be bound to the chosen numerator.
4. A theoretical design specification can underpin an engineering roofline assumption. A strict no-faster-than certificate additionally needs its SKU/configuration applicability and any allowed positive error/boost envelope. No arbitrary multiplier is substituted for that evidence here.

The older Huawei Da Vinci architecture paper describes Cube MAC geometry but predates910B; it is not used as a910B3 maximum-issue certificate. PlatformAscendC documentation exposes hardware-information queries; an API's existence or a tiling parameter is not itself a proof that its value upper-bounds every execution.
Primary documentation:
https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00059.html

## 4. Smallest promising numerator candidates

### A. Conventional routed expert evaluation

One ordinary full routed expert pair for one fresh token has:
w_pair = 2(4096*4096 + 2048*4096) =50,331,648 standard operations.

For one ordinary Target prediction row:
w_target_routed =43*6*w_pair =12,985,565,184 standard operations.

Source/configuration: /data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json; ASCEND/models/deepseek_v4.py:435-461; scripts/loop077_device_capture_analyze.py:11-12. Run375/380 and Run403/405 support current shape/routing counts.

This is a positive exact arithmetic count within the conventional dense top6 evaluation class. No weight/HBM or Draft accounting is needed to USE that count once compulsory membership and timing are certified. But it is not a proven operation-complexity lower bound for every semantically equivalent algorithm: result reuse, exact structured transforms, changed precision with proved equivalence, or other permitted algorithmic reductions may change standard-operation work. Frozen weights and correct outputs alone do not make2MNK a universal lower bound.

Currently no matching certified W4A8 C^+ exists. A measured GMM peak is not a substitute.

### B. BF16 wo_a subset to avoid the INT8/FP16 mismatch

ASCEND/models/deepseek_v4.py:775-782 defines groupwise wo_a with4096 input values and1024 output values per group. ASCEND/attention/context_parallel/dsa_cp.py:1374-1391 uses transpose_batchmatmul before wo_b on the inspected910B path. Run370 records43 M96_K4096_N1024 transpose-batch-matmul tasks per rank.

One group for ONE fresh retained prediction row, at ONE required Target layer, has a conventional count:
w_one_group =2*4096*1024 =8,388,608 operations.
All8 groups at that layer:67,108,864 operations.
This avoids counting replicated WKV/Compressor work. The saved rank0 latest Run246 ASCEND_PROFILER_OUTPUT/kernel_details.csv directly reports aclnnTransposeBatchMatMul inputs [96,1,4096] and [1,4096,1024], both DT_BF16, and output [96,1,1024] DT_BF16. This pins the observed dtype; it does not establish the maximum device BF16 capacity or algorithm-independent necessity.

This is the best small FLOATING-POINT candidate, but exact semantics still require a certificate:
- the selected group contributes to a retained observable result and cannot be supplied by a preexisting cached result;
- the admitted algorithm class performs this dense projection (rather than eliminating/factoring it or replacing a full-logit requirement with an equivalent decision procedure);
- its actual BF16 input/output and accumulation interpretation match a certified hardware capacity convention;
- the counted work is confined to engines covered by C^+, or those additional engines are included in C^+.

Neither "kernel observed" nor "model has a dense matrix" proves those clauses for every allowed optimized Runtime.

For scale only, IF3.008e15 operations/s were certified as a matching full-machine ceiling AND one such group had to be newly evaluated for each external token, the conditional algebra would give358,581,542.96875 tok/s. This intentionally loose result is NOT a promoted bound: both IF clauses remain uncertified. It demonstrates that a tiny subset can suffice; demanding all43 layers or all49,152 tokens is unnecessary for mere finiteness. Even one certified in-window projection for the whole workload would yield the finite but much looser N*C^+/8,388,608.

### C. Weight/HBM, communication and publication alternatives

- An active packed expert storage set is not necessary physical HBM traffic. Current cache residency, preloading and alternate representations prevent using stored bytes as W^- without a boundary certificate.
- Run414's dense transport cut counts explicitly freeze current raw materializations and forbid recomputation/zero elision; they are not model-semantic W^-. The new vendor HCCS design number does not repair that numerator.
- External publication necessarily conveys an answer, but this benchmark uses local HTTP paths in the saved scripts. A25/50-Gbit NIC inventory cannot be assigned as its bottleneck. Token count is also not an invariant minimum number of transmitted raw bytes under all representations. A host output-copy rate would need its own interface/boundary and maximum-rate certificate.
- A single comparison/activation appears intuitively unavoidable, but assigning it to a particular NPU instruction and claiming one instruction per output without a model-to-engine proof is not a strict bound.

## 5. Exact semantics and Product timing

runtime/fixed_acceptance.py:27-35 and runtime/greedy_accept.py:42-55 verify greedy Target predictions. For a leading matched draft prefix of a tokens, raw accepted count=a+1 includes recovery or bonus; it is not a drafts-only count.

For a particular new output prefix, a standard verification trace supplies a corresponding Target prediction position. Overshoot clipping, already-published handoff output, cache reuse and common-prefix reuse prevent automatically multiplying a per-row count by49,152 for ANY alternative implementation.

Run419 explicitly marks external pre-handoff p_i unverified despite internal zero counts. This does not prevent every positive subset proof: proving only ONE fresh necessary in-window projection would be enough. It does prevent using the stronger all-token multiplier without its join. Likewise, source algebra for one logical projection does not need all43 propagated row labels; an exact one-row consumer lineage and freshness witness would suffice.

A measurement can establish this witness for a frozen execution class. Extending to arbitrary future algorithms requires a stated invariant/algorithm restriction, not more samples of the current code.

## 6. What is minimally missing

The smallest useful new deliverable is a TWO-CLAUSE subset certificate, not a full resource census:

W clause:
- name one required logical wo_a projection/group (or another primitive);
- pin dtype and algebra;
- identify at least one fresh live prediction row, its required consumer and benchmark time boundary;
- explicitly scope the implementation class in which its count cannot be removed/reused/replaced.

C clause:
- bind these actual910B3 boards/configuration to an authoritative maximum rate for that work, using either a vendor per-chip rate with maximum clock/error envelope or per-engine issue width times maximum clocks;
- account for all engines the class permits to perform the work, or prove the subset is confined to the bounded engine;
- retain MAC/operation, precision and sparsity conventions.

If conventional dense NPU evaluation is accepted as the stated bound class, the highest-value remaining concrete certificate is the SKU-bound BF16 maximum-throughput/clock-and-issue document. It can be read/configuration evidence; a new service experiment is not inherently required. The existing fresh-row lineage work can supply the W clause, without requiring complete Draft/KV/HCCL coverage.

If the claim must cover EVERY mathematically equivalent model implementation, the W clause remains a separate proof obligation. No single bandwidth or GEMM microbenchmark can solve it. A faster observed run may refute a proposed C^+, but a finite set of slower runs cannot certify that C^+.

## 7. Capacity categories and recommendation

- Theoretical hardware C^+: suitable for a very loose Resource/Product ceiling when the subset and SKU bindings are certified. It need not be attainable on model shapes.
- Attained isolated engineering service: a feasibility sample for that implementation. It does not upper-bound hardware capacity.
- Mixed compute/HBM/HCCL occupancy: needed to tighten or predict a practically attainable limit, not to prove the first valid loose ceiling.
- Complete Product DAG and repeated formal E2E: required for claims of attainable optimized performance, not prerequisites for the simple subset inequality.

Recommended status:
strict_algorithm_semantic_product_tps_upper = null
conventional_dense_subset_bound = conditional, awaiting W/C certificate
complete_achievable_product_bound = null
Current Formal remains the separately established measurement.

Confidence: high in the subset theorem and rejection of measured peaks as C^+; high in saved actual910B3 identity and primary-document provenance; medium in wo_a as the smallest useful engineering subset; insufficient for mandatory exact-algorithm work and matching actual-SKU capacity to promote a numerical ceiling.
