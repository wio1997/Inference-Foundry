# GLM-5.3-Flash W8A8：910B 上的 SGLang / vLLM P/D 对比

本分支按当前任务持续归档测量、机制、源码与失败证据。**性能对齐尚未完成，Sinkhorn 融合的端到端收益尚未验证。** 所有时间为 UTC；日期为 2026-11-17。

| 编号 | 对象 | 一句话结论 | 关键数字 | 状态 | 入口 |
|---|---|---|---|---|---|
| 001 | 固定镜像、W8A8、TP8/EP8/MTP3 | vLLM 四档、原 SGLang 32K 的冷热原始测量已保存 | SG 32K 冷/热输出 128.3895 / 126.3610 token/s | 有测量；最终公平对齐未验收 | [测量与边界](docs/001-baseline.md) |
| 002 | 当前双框架 P/D profile | 每框架各16个rank有真实模型NPU事件 | SG 每8步出现28,080次除法；窗口差异已列明 | 诊断证据；不能作墙钟份额 | [profile地图](docs/002-profile.md) |
| 003 | 20轮 Sinkhorn 融合 | 数值、图回放与完整P/D检查通过 | 27数值、4图、5适配器、7个P/D用例 | 端到端收益未验证 | [机制与源码](records/points/GLM-OPT-0002/point.md) |

正式与预热分账；未通过稳定门槛的预热不改写为正式结果。权威数据在 [metrics.json](records/metrics.json)；[逐Run索引](records/runs.json) 保留种类与裁决。

[环境与合同](environment/contract.json) · [离线验档](scripts/validate_archive.py) · [原始数据与脱敏](docs/003-evidence.md)
