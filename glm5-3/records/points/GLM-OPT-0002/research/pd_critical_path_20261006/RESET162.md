# Reset162 — source-supported KV capacity correction

Upload gate is cleared by the user's two-host completion statement. Branch/head remains `glm5-3-autonomous-20261001` / `df534e5d`; rules `GLM-RESEARCH-RULES-v2`, Current=None, active PERF_KEEP empty. Product contract and conditional H3 from Reset161/160 remain unchanged. The immediate blocking gap is native initialization, before a PD request can execute.

Run245 actually loaded the new model and MTP weights on all16 ranks of each role. The P log prints46.6157GB and D46.5611GB. Both then rejected the configured1GiB KV budget: native `_check_enough_kv_cache_memory` needs1.53GiB for at least one144384-token request. Exact installed sourceSHA `0ab5385925f660395612e3f69030cad3b18c2e0bd6fb61d3ff43072a6cefb74f`, lines751–787, compares `get_needed_memory()` against available bytes. This is a capacity validation failure, with no executed P→D probe or performance result. Sol read both decisive errors and all32 weight-load records. Run245 `execution_summary.json` references actual raw hashes, failed phase, cleanup and controller. Old gateway is stopped; no replacement public contract is accepted.

REUSED: Run245 establishes weight loading/layout compatibility, safe six-key CANN environment and exact failure. No repeated header scan or old PID retirement is useful. Fresh read-only observation shows both containers running, all32 NPU owners empty, relevant API/Mooncake ports unbound. Existing controller/phase/ACL wrappers stay unchanged; original source/spec/raw remain preserved.

Only Run246: change KV budget to2GiB (above the native rounded1.53GiB requirement), preserve144384 context and all model/MTP/DP1/TP16/PP1/EP16/DCP16/role flags. This fixes the known check; actual allocation, warmup and PD behavior remain unknown. It does not prove four simultaneous maximum-context requests or SLA capacity. Run-scoped startup checks fresh zero owners, ports, controller identity, hostboot and frozen source hashes before launching. Environment provenance comes from actual Run245 snapshots; no stale PIDs are signaled.

Hypothesis(functional): correcting this one known KV budget deficiency allows both roles to initialize and the unchanged native P helper→KV metadata→D path to produce32 effective D tokens.

Distinguishing evidence: native allocation/warmup/health and served-model identity; actual P metadata and D cached-token/token-ID/output evidence; actual failure and owned cleanup. H3 timing, full API coverage and performance remain unaccepted.

Decision table: initialization fails→read the first specific new cause, preserve raw, clean only owned processes; roles ready but probe fails→retain healthy roles and diagnose that request; valid native PD output→continue full dynamic API lifecycle validation and the conditional H3 diagnostic. No configuration sweep, local-D fallback or kernel/source patch. Prior Astra Review remains applicable; this capacity correction is not an architecture fork.

本阶段没有新增代码级性能 KEEP。完整5.3 PD E2E Gain=unknown；Current=None。
