# 执行日志

- `2026-09-21T05:07:33Z` Loop 已冻结。下一步：Distinguish EVENT_WAIT device streams and event producer tasks for outer draft and moe_forward_shared waits, then align them with fused_moe.py event callsites.

- `2026-09-21T06:05:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `tp0-event-topology-20260921`（profile）。

- `2026-09-21T06:05:32Z` Run `tp0-event-topology-20260921` 记录为 `pass`；正确性为 `not-applicable`。Outer 53-54 ms waits occur once per step on streams 42/43 and immediately gate MEMCPY_ASYNC, identifying auxiliary copy streams. MoE stream36 initial before_routed wait has 25.734 ms median because it idles between layer invocations until current hidden states are ready; intra-call waits are only 0.191, 0.051 and 0.011 ms median. Default stream47 final waits are about 0.00002 ms, so shared-expert overlap does not expose a material main-stream stall.

- `2026-09-21T06:05:32Z` 主控结论为 `REJECTED`。Exact stream neighbors falsify a material main-path MoE wait. The two 53-54 ms outer waits gate MEMCPY_ASYNC on auxiliary streams 42/43. The apparent 25.734 ms MoE wait is shared stream36 waiting at the next layer before dynamic quant; default stream47 waits for shared output are about 0.00002 ms. Intra-call synchronization medians are 0.191, 0.051 and 0.011 ms, below baseline noise.
