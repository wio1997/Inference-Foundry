# GLM-5.3-Flash 独立对话接手提示

你只负责项目`ascend-framework-comparison/glm53-flash/`，目标：先建立Flash的vLLM文本/PD参考，再完成SGLang Flash混合状态/KDA与1P1D适配，后单图/多模态和性能。

先读**本目录**AGENTS.md、HANDOFF.md、RESOURCES.json、PLAN.md、STATE.json。需要性能时读本目录BENCHMARK_CONTRACT.md，需要实际操作时读本目录EXECUTION_CONTRACT.md。不要把父目录总HANDOFF/总合同或另一个模型目录作为必读依赖；两个对话没有任务、参数、进度、合同或验收依赖。

使用GPT6.1 Sol High做主控；所有实际构建、传输/镜像导入、部署/服务启停、NPU探针、模型请求、压测/profile均经用户Mac本机已配置DeepSeek的ZCode，主控审查具体脚本/目标/回退并冻结SHA，精确launcher一次。不要读取provider配置或凭证，不恢复定时、不创建或消息另一对话。

启动前校验模型身份`glm5_next` / `Glm5NextForConditionalGeneration`、路径`/data/wio/models/GLM-5.3-Flash-w8a8`及量化/索引/模板实际身份，固定本项目的镜像/SDK和数据。只管理RESOURCES列出的本项目节点及本次自有任务，读取实际状态/进程/锁，不重复既有任务。资源批次就绪不代替模型正确性；不改系统SDK/驱动/BIOS/时钟，不动无关作业。

每次运行保持project_id=glm53-flash，使用本项目独立namespace、配置、锁、Meta/provider、数据、原始失败和结果目录。各自合同独立版本化，改本项目参数不改另一项目。不得把另一模型的吞吐、缓存布局/命中常数、模型层数或PID当本项目事实。

资源：vLLM14P/15D，SGLang46P/47D。先Flash单机文本/混合状态正确，再本项目真实1P1D、图/MTP/缓存；多模态只读本目录MULTIMODAL_TEST_PLAN.md及multimodal-matrix.json。原模型的MLA/MemCache补丁不能替代FlashKDA/conv状态适配。

P→D数据面强制使用**SDMA**，同时适用于本项目vLLM-Ascend与SGLang；不允许自动回退RDMA传输模型状态。按本目录[SDMA合同](PD_TRANSPORT.md)核配置、实际加载路径和真实请求传输证据；TCP控制/握手与PD数据面分开。此项为用户指定要求，尚不是运行验证结论。
