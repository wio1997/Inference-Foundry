# 原 GLM-5.3 独立对话接手提示

你只负责项目`ascend-framework-comparison/glm53/`，目标：原模型两套2P2D：先新SDMA vLLM基线，再SGLang同模型/精度基线与profile驱动优化；W8A8C8另列实验。

先读**本目录**AGENTS.md、HANDOFF.md、RESOURCES.json、PLAN.md、STATE.json。需要性能时读本目录BENCHMARK_CONTRACT.md，需要实际操作时读本目录EXECUTION_CONTRACT.md。不要把父目录总HANDOFF/总合同或另一个模型目录作为必读依赖；两个对话没有任务、参数、进度、合同或验收依赖。

使用GPT6.1 Sol High做主控；所有实际构建、传输/镜像导入、部署/服务启停、NPU探针、模型请求、压测/profile均经用户Mac本机已配置DeepSeek的ZCode，主控审查具体脚本/目标/回退并冻结SHA，精确launcher一次。不要读取provider配置或凭证，不恢复定时、不创建或消息另一对话。

启动前校验模型身份`glm_moe_dsa` / `GlmMoeDsaForCausalLM`、路径`/data/wio/models/GLM-5.3-w8a8`及量化/索引/模板实际身份，固定本项目的镜像/SDK和数据。只管理RESOURCES列出的本项目节点及本次自有任务，读取实际状态/进程/锁，不重复既有任务。资源批次就绪不代替模型正确性；不改系统SDK/驱动/BIOS/时钟，不动无关作业。

每次运行保持project_id=glm53，使用本项目独立namespace、配置、锁、Meta/provider、数据、原始失败和结果目录。各自合同独立版本化，改本项目参数不改另一项目。不得把另一模型的吞吐、缓存布局/命中常数、模型层数或PID当本项目事实。

资源：vLLM P30/31 D43/45，SGLang P34/35 D36/41，各2P2D、每角色32die，先重算并行/KV/MemCache配置。32/33补充W8A8C8与rc1资源已齐，但P/D角色未分配，不自动替换W8A8基线或参与主组算力/缓存。每个精度变体单独run/基线。无需读取多模态方案。
