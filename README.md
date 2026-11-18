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

[vLLM G4 首轮32K正式冷热](docs/007-vllm-G4-first-32K.md)

[vLLM G4 首轮64K正式冷热](docs/008-vllm-G4-first-64K.md)

[vLLM G4 首轮128K正式冷热](docs/009-vllm-G4-first-128K.md)

[同一vLLM G4运行时四档首轮冷热完整基线](docs/010-vllm-G4-four-shape-first-cold-hot.md)

[G4 profiling真实设备窗口验收与D触发修正](docs/011-vllm-G4-profiler-device-gate.md)

[G4 同一采集批次十六 rank 真实模型设备窗口](docs/012-vllm-G4-all16-model-profile.md)

[同机 SGLang 迁移、控制器目录失败及修正](docs/013-matched-host-SG-migration.md)

[同机 SGLang 实际加载、完整图与 PD 功能门槛](docs/014-matched-SG-real-PD-readiness.md)

[同机 SGLang 首遍完整32K/C16/1024未计分预热](docs/015-matched-SG-first-full-warmup.md)

[同机 SGLang 七轮独立预热与编译门槛](docs/016-matched-SG-seven-warmups.md)

[同机 SGLang 32K 首轮正式cold/hot及待定位差异](docs/017-matched-SG-first-32K-cold-hot.md)

[同机 SGLang 与G4匹配输入的十六rank模型profile](docs/018-matched-SG-all16-model-profile.md)

[同机 SGLang 64K 五轮独立预热](docs/019-matched-SG-64K-five-independent-warmups.md)

[匹配 profile 的机制差异与高潜力候选](docs/020-profile-mechanisms-and-priority-candidates.md)

[同机 SGLang 64K 首轮正式冷测与热测](docs/021-matched-SG-first-64K-cold-hot.md)

[首次128K预热失败与原生超时修复](docs/022-first-128K-warmup-failure-and-timeout-repair.md)
