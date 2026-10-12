# 新会话恢复

核对实际 branch/HEAD 与未提交改动，再读 [AGENTS](../AGENTS.md)、[HANDOFF](../HANDOFF.md)、[MISSION](../MISSION.md)。不从本文件触发现场恢复、代理 CLI 演练或模型调用。

```text
接续 Inference-Foundry/glm5-3 的 GLM-EXECUTION-RESEARCH-v3。
目标是独立执行计划研究，SGLang/vLLM 是外部对照与可选集成。
先核对 Git，再读 AGENTS、HANDOFF、MISSION 和 GLM-PLAN-0001。
现场调优仍暂停，只做本地实现和离线分析。
从已有证据选真实片段，区分必要依赖与框架顺序，列出保真缺项。
不把 synthetic 成本/CPU replay 当成 GLM/NPU 性能，不恢复旧 next_action。
```

代码改动后用 [README](../README.md) 的本地测试。证据足够则复用；设备 fixture 缺失如实保留，并继续不依赖它的代码与分析。
