# DP / PP architecture comparison review — 2026-10-06

Trigger: the user challenged why the recovered deployment uses PP and whether DP has greater product value. This is a major architecture comparison under GLM-RESEARCH-RULES-v2. Independent Astra review reused the existing Challenger; no service operation, benchmark, code patch, or candidate activation. Current=None; active PERF_KEEP empty. H1 remains scoped REJECT; H2 admission remains INCONCLUSIVE/PARKED.

## Evidence personally checked by Sol

- Fresh native argv: 166 DP1/TP8/PP2/DCP8/K2, 167 DP1/TP4/PP4/DCP4/K1; two independent full-request domains, no EP or KV connector. The existing HTTP entry already offers request-level replica parallelism.
- Actual installed `/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/config.py`, SHA256 `c4cfa8a9775144058d581162aaed9bc4441c07fcb1f51e8eb1545e13cc08f6ad`, lines1113–1121 and1204–1244: MoE flattens DP×PCP×TP. With EP enabled experts are distributed across this group; with EP disabled the MoE TP group is still flattened. DP2/TP8/EP16 is not two independent eight-device complete models. Attention, KV, Graph, MTP and runtime peak memory still need actual validation.
- Actual Ascend platform.py SHA256 `8ef33020bfb0959e989f0f3618d54eeea6aaa8a9c2788b1da7f41af3e230c7ff`, lines1521–1527: the active SFA replicated-indexer class requires DCP=TP. This constrains a layout; it does not require PP.
- Historical Run38 manifest and reduction_brief: DP4/TP8/DCP8/EP32/PP1, 32 physical devices,37 completed requests/35546 effective outputs. Original `runs/GLM-RUN-0038/DP_run38_0.log` SHA256 `7b1f9e109558c15ffe0713ce3346010da037f6995871401c900bb0078d450119`: line50 explicitly records TP8/PP1/DP4/DCP8; line1235 records TP0 weights27.02GiB, activation1.73GiB, non-torch2.57GiB, Graph0.93GiB, KV17.69GiB. All four original log hashes match the existing reduction_brief. This is genuine historical fit evidence for EP32, not fit proof for EP16 or current K2.
- Run38 dynamic90.4787TPS has different cache, arrivals/concurrency, MTP and deployment conditions from current PP evidence. It cannot rank architectures or establish product Gain. Current PP device evidence and its limitations remain in H2_DEVICE_DECISION.md: compute-stage overlap about3.57% of the observed window, while the roughly34ms receive includes necessary target/MTP peer dependence. No measured removable34ms or predicted DP gain.

## Independent Astra result

DP deserves priority as a comparator, but DP performance has not been proved superior to PP. PIVOT research priority to a PP1 deployment comparison; KEEP DP candidate value only, not PERF_KEEP or deployment activation.

Known: current service already has two complete-request replicas. DP2/TP8/EP16 shares experts over16 devices. Historical DP4/TP8/EP32 actually fit and completed requests. Native MoE DP requires expert communication, aligned forwards, dummy work on idle ranks and synchronization; no2× throughput guarantee.

Unknown: local DP2/TP8/EP16 actual fit and complete correctness; equal-contract throughput/tail latency; whether removing PP dependencies beats added TP/EP communication and DP idle synchronization. The largest missing architecture evidence is a matched product comparison; the largest code-removable critical-path time remains unknown.

Top3 questions: (1) local DP2 fit, long-context KV, Graph/MTP and full features; (2) saved PP dependencies versus added TP/EP and DP synchronization; (3) SLO-qualified capacity under uneven arrivals for coupled DP versus independent TP16 replicas.

Minimal distinguishing validation: choose one question. To ask whether PP should remain, two independent TP16/PP1 domains are a more direct comparator. To ask whether local DP2 should be adopted, pass its fit/correctness gate, then compare with independent TP16. Hold32 physical devices, model, contract, effective counts, arrivals, cache conditions and sampling/MTP comparable. If implementation constraints prevent matching, compare complete deployments and disclose the differences; do not attribute the result to DP alone. These are alternatives, not an execution queue.

Stop: inertia-driven PP optimization; transferring EP32 memory numbers to EP16; ranking unmatched historical TPS; configuration Gain as code KEEP; claims that DP must win, PP is necessary, or capacity doubles.

## Sol decision

Accept all findings. DP is a high-value architecture comparator, not a proved performance winner. PP is not a default requirement or an accepted optimal baseline. Prioritize source/evidence preparation for one PP1 comparison rather than implementing the parked PP admission candidate. No architecture is selected for deployment and no performance hypothesis/Run is activated by this review. A fresh supporting Reset, exact candidate contract, correctness and decision table are required before a new device Run; matched noise-aware complete E2E and all product acceptance requirements remain necessary for Current/PERF_KEEP.

本阶段没有新增代码级性能 KEEP。True complete E2E Gain=unknown; Current=None.
