# GLM-5.3 W8A8C8 / vLLM-Ascend / Ascend 910C

分支：`codex/glm5-3-vllm-optimization`。从原 SGLang 研究分支 `codex/glm-execution-plan-reset` 的 `6b44d8b2755f9527b3066fdd7a014ed5cdc062bb` 建立，保留此前资料。

用户在现有 Ascend 环境中观察到 vLLM-Ascend 更稳定，因此本线优先用它承载研究与集成。研究对象是既有算子之上的计算、状态和执行组织；框架名称不决定性能上限。SGLang 提供机制参考和必要的差异对照，不要求同步维护两套完整实验系统。

| 入口 | 用途 |
| --- | --- |
| [MISSION](MISSION.md) | 服务目标、当前范围与探索空间 |
| [AGENTS](AGENTS.md) | 少量协作规则与设备边界 |
| [HANDOFF](HANDOFF.md) | 已知事实、参考证据、未知项与下一交付 |
| [PLAN](PLAN.md) | 按需参考：机制研究、真实片段和实验周转 |
| [硬件地图](HARDWARE_MAP.md) | 首轮实测拓扑、软件/资产身份、SDMA 测量边界与未知项 |

先补齐影响首轮判断的硬件地图与正确基线，从值得检验的机制决定真实片段边界，再对比不同执行组织。减少参与实验的模型范围，保留可能改变结论的形状、状态和通信条件。正式目标保持 PD 分离；32、33 能否组成 1P1D 尚待验证，局部实验不替代完整 PD 验收，也不要求每次局部尝试都重新加载完整模型。

2026-10-17 已恢复 32/33 现场工作，完成首轮只读环境/拓扑/权重凭据核查与 33 单 die 无权重功能检查；用户要求数据搬运全部 SDMA，并授权释放两台原实验占用。尚无完整模型/PD 正确基线或推理性能收益。当前状态及资源归属见 HANDOFF。

按需复用 [SGLang 部署与测试资料](../glm5-3-sglang/README.md)、[旧研究审计](../glm5-3/docs/RESEARCH_RESET.md) 和 [补丁证据](../glm5-3/patches/20261007/README.md)。它们的模型、精度、拓扑和负载条件不自动成为本线默认值。

续接提示：`继续 glm5-3-vllm；按 AGENTS.md，从 MISSION.md 和 HANDOFF.md 恢复。以稳定 vLLM-Ascend 路径为参考，优先用真实片段研究执行机制；设备操作仍按当前许可边界执行。`
