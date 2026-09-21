# Loop 报告：loop-014

- 结论：`PIVOTED`
- 决定时间：`2026-09-20T21:04:13Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

No supported >=5% mixed TPS candidate emerged. The 8 metadata builders are group-specific and shared local/ratio state is already cached. First builder timing may include required device synchronization; Loop011 all-step QLI substitution did not improve mixed throughput. Avoid changing semantics merely to reduce inclusive host spans.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_loop014_prepare_trace/run2/builder_summary.json`
- `/data/wio/Inference_Foundry/evidence/20260920_loop014_prepare_trace/run1/stage_summary.json`

## 下一步

Restore diagnostic patch and baseline-flag service, then Loop015 examine cold prefill ScatterNdUpdateSk/Compressor and HCCL critical path for a falsifiable improvement.
