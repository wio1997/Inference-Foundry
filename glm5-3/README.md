# GLM Extreme

为GLM构建完整功能的专用推理框架，研究现有算子下多机多卡、动态负载及可变PD的调度极限。当前用GLM-5.2、两台910C；版本/资源口径见MISSION。

**Agent启动只读[AGENTS](AGENTS.md)+[HANDOFF](HANDOFF.md)+活动点/Git状态。**

| 入口 | 按需用途 |
|---|---|
| [MISSION](MISSION.md) / [PLAN](PLAN.md) | 目标/合同、极限标准与逼近逻辑 |
| [RECORDING](RECORDING.md) / [优化点索引](records/points.jsonl) | Git提交、Run证据及裁决 |
| [REUSE](REUSE.md) / [复用索引](records/reuse.jsonl) | 旧成果适用性、失败原因与去重 |
| [研究索引](docs/README.md) | GLM特性、源码、prof、历史迁移与模型成本 |

交给执行Agent的短prompt：

```text
在glm5-3/按AGENTS.md和HANDOFF.md接续。恢复实际环境与Current，先查相关历史，再从最高价值Gap/未知自主研究、重构和验证。按RECORDING记录优化点、代码与Run；当前两台可行域内收敛。资料按需读，不重放历史，不等待未来硬件。
```

初始发布不包含性能改动或已运行实验。当前第一个点是现场/Current恢复。
