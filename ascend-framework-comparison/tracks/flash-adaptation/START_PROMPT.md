# 可交给独立对话的接手提示

在 wio1997/Inference-Foundry 的 codex/ascend-sglang-vllm-comparison 分支，负责 ascend-framework-comparison/tracks/flash-adaptation，目标为「GLM-5.3-Flash：vLLM 参考与 SGLang 适配」。先读公共 AGENTS.md、HANDOFF.md、PLAN.md、BENCHMARK_CONTRACT.md、RESOURCES.json 和本轨道 STATE.json，再检查实际状态避免重复运行。

先读原生混合状态语义、固定同一 Flash W8A8 权重和官方版本。14/15 是 vLLM 1P1D 参考，46/47 是 SGLang 目标；先单机文本正确，再真实 PD，再图/MTP/缓存。

用户已明确不要共享前缀：C 个独立 74400-token 前缀，各自预热后连续复用四轮，全新后缀、完整 80000/600、4C 条完整输入各不相同。预热完全排除性能统计，保留自然缓存，不强制预热后清 L1/L2。历史 RDMA+共享前缀数据只溯源，不能作本轮对照。

现场无外网，Mac 下载并校验后离线上传。实际构建/部署/模型和性能执行通过用户本机 ZCode 的 DeepSeek，由主控审查冻结精确 launcher、只执行一次并报告真实 receipt；保留已有恢复环境，不改系统 SDK/驱动/BIOS/时钟，不启动定时任务。先核验设备占用与用户授权范围，不触碰其他轨道/无关作业。没有访问地址时先准备离线工作，不猜 IP。

正确性通过才建立完整基线；每轮保存真实 argv/源码/SDK/权重身份、原始失败、缓存来源、实际运行/排队和端到端时间。只在本轨道提交改动；公共合同有分歧先记录并由主控统一，不私自改计量。当前阶段以本用户请求和公共状态为准，不能重放旧归档中的一次性 launcher。
