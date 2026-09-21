# 执行日志

- `2026-09-21T10:07:05Z` Loop 已冻结。下一步：Audit the three-layer forward for graph-unsafe state/HCCL/KV mutations and run the smallest exact-shape capture feasibility probe; do not enable unsupported full proposer graph in production.

## 2026-09-21T11:07:00Z

- Source audit: legacy DSpark proposer disables graph unconditionally; v2 runner contains a registered AscendDSparkSpeculator with an Ascend query graph manager.
- Started minimal startup/correctness probe with VLLM_USE_V2_MODEL_RUNNER=1 and enforce_eager=false; all frozen model/parallel/scheduler settings unchanged.
- API PID 891755 loading. No correctness or performance result yet.
