# 执行日志

- `2026-10-30T15:31:28Z` Loop 已冻结。下一步：lazy-import预检已通过；审核owner fence/watchdog/恢复原服务的166诊断控制器，随后执行有限模型补采。

- `2026-10-30T15:44:29Z` 为用例 `c8-input32k-output1k-prefix90-166` 创建 Run `run-20261030T154429Z`（test）。

- `2026-10-30T16:03:15Z` Run `run-20261030T154429Z` 记录为 `error`；正确性为 `invalid`。诊断在启动profile_run被取证脚本中未证实的M<=512/E8or24断言中止；精确M/E未记录，不编造。未产生有效请求计数。原服务已由控制器恢复。

- `2026-10-30T16:03:15Z` 主控结论为 `PIVOTED`。合成graph helper与全负载参照有效，但模型取证入口的额外512行断言不成立为通用调用前提；源selector可无num_tokens检查进入custom FUSED。首轮未得到有效模型计数，原服务已恢复。修订形状域与ownership非原子采样处理后建立派生Loop。
