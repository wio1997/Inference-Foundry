# 当前交接 — 2026-10-12

**当前阶段：独立执行计划实验入口已建立，真实 GLM 片段尚未接入。设备测试继续暂停。**

- 规则：`GLM-EXECUTION-RESEARCH-v3`，见 [AGENTS](AGENTS.md)。目标与身份缺项只在 [MISSION](MISSION.md) 维护。
- 本轮完成：修正过期入口；审计并保留历史收益；加入独立 problem/plan、资源约束估计和 CPU reference replay。它不调用 SGLang/vLLM，也不是 Ascend 推理引擎。
- 产品 Current：**unknown / 未验收**。H6/H5 是各自历史合同内的研究收益，不能相加或搬到未知现场。见 [manifest](patches/20261007/manifest.json) 与 [审计](docs/RESEARCH_RESET.md)。
- 新代码证据类别：本地工具正确性 / synthetic 成本示例；**本轮没有新增 GLM 性能收益或 NPU Run**。
- 当前研究点：[GLM-PLAN-0001](studies/GLM-PLAN-0001.md)，选择并校准真实片段。第一问题是“旧证据能否提供可回放、对干预响应保真的真实片段”，不是预定某个 patch、MoE 重叠或框架重写必然有效。
- 下一步：按研究点字段，从历史源码/profile 定向提取必要依赖、实际输入条件和已知干预；缺失保持 unknown。恢复设备授权后才采集缺失 tensor/路由和短设备窗口。
- 不恢复旧 controller、PID、部署默认值或自动 CLI 模型路由；`legacy/`、旧 `docs/research/`、`records/` 和发布补丁均为带日期历史。

本地入口与测试见 [README](README.md)。后续“完成”应说明是工具完成、机制完成还是产品验收完成。
