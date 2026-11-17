# GLM-5.3-Flash W8A8：910B 上的 SGLang / vLLM P/D 对比

本分支按当前任务持续归档测量、机制、源码与失败证据。**性能对齐尚未完成；32K首次冷热观测相对旧SG输出吞吐高16.01%/18.25%，独立归因和可重复收益仍待验证。** 所有时间为 UTC；日期为 2026-11-17。

| 编号 | 对象 | 一句话结论 | 关键数字 | 状态 | 入口 |
|---|---|---|---|---|---|
| 001 | 固定镜像、W8A8、TP8/EP8/MTP3 | vLLM 四档、原 SGLang 32K 的冷热原始测量已保存 | SG 32K 冷/热输出 128.3895 / 126.3610 token/s | 有测量；最终公平对齐未验收 | [测量与边界](docs/001-baseline.md) |
| 002 | 当前双框架 P/D profile | 每框架各16个rank有真实模型NPU事件 | SG 每8步出现28,080次除法；窗口差异已列明 | 诊断证据；不能作墙钟份额 | [profile地图](docs/002-profile.md) |
| 003 | 20轮 Sinkhorn 融合 | 32K首次正式冷热观测改善，最终对齐未完成 | 输出148.9474 / 149.4161 token/s | 独立归因/重复性待验证 | [首次端到端观测](docs/004-first-end-to-end.md) |

正式与预热分账；未通过稳定门槛的预热不改写为正式结果。权威数据在 [metrics.json](records/metrics.json)；[逐Run索引](records/runs.json) 保留种类与裁决。

[环境与合同](environment/contract.json) · [离线验档](scripts/validate_archive.py) · [原始数据与脱敏](docs/003-evidence.md)

[G43 完整模型融合 profile](docs/005-fused-model-profile.md)

[vLLM G4 32K 未计分预热](docs/006-vllm-G4-warming.md)
