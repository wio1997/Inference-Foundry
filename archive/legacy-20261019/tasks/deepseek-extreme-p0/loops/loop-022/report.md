# Loop 报告：loop-022

- 结论：`REJECTED`
- 决定时间：`2026-09-21T06:05:32Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`not-applicable`
- 因果结论：`falsified`

## 决定依据

Exact stream neighbors falsify a material main-path MoE wait. The two 53-54 ms outer waits gate MEMCPY_ASYNC on auxiliary streams 42/43. The apparent 25.734 ms MoE wait is shared stream36 waiting at the next layer before dynamic quant; default stream47 waits for shared output are about 0.00002 ms. Intra-call synchronization medians are 0.191, 0.051 and 0.011 ms, below baseline noise.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_loop021_device_parent/event_wait_streams_tp0.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop022_event_dependency/wait_stream_neighbors_tp0.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop022_event_dependency/moe_wait_stages_tp0.json`

## 下一步

Test whether seven-token drafting is overlong for measured 3.54-3.64 token advancement by benchmarking a bounded speculative-length sweep under the frozen mixed workload.
