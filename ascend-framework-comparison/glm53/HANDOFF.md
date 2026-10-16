# 原 GLM-5.3 当前交接

本项目独立于另一个模型的对话。此前混合总入口已停用；默认只读取本目录。节点与预期模型身份以[本项目RESOURCES](RESOURCES.json)为准，具体源/目标证据在[ASSETS](ASSETS.md)。本项目资源准备由本对话完成，接手者可能已在现场推进，启动前核实际进程/锁，不把本文资源记录当实时服务状态。

原模型两套2P2D：先新SDMA vLLM基线，再SGLang同模型/精度基线与profile驱动优化；W8A8C8另列实验。

权重为`/data/wio/models/GLM-5.3-w8a8`，glm_moe_dsa / GlmMoeDsaForCausalLM。候选Docker实际ID均已独立验证，尚无本资源任务产出的新模型服务/性能基线。源码快照与旧资源批次目录只作证据，不是要恢复的服务。当前新运行namespace`/data/wio/inference-foundry/glm53`为计划，未在这次文档拆分中创建。

GPT6.1 Sol High主控、本机ZCode DeepSeek执行真实动作；详见[本项目执行规则](EXECUTION_CONTRACT.md)。不恢复定时，不自动创建/消息另一对话。只改本项目的配置、脚本、数据、STATE、结果和run；不需要另一模型跑通、同意或改合同。

下一步：读取本项目PLAN，核独占与image/模型身份，经本机ZCode冻结脚本开始本模型正确性门控；完整通过后用本项目BENCHMARK_CONTRACT建立基线。预热不计吞吐，不共享长前缀，失败全保留，实际缓存与运行/排队以证据为准。

原模型主两侧各2P2D：vLLM P30/31 D43/45，SGLang P34/35 D36/41。每角色32die，重新冻结TP/attentionDP/EP、KV/MemCache与图/MTP，不能照搬旧4P EP64/TP64。32/33 C8+rc1已独立验收完成，P/D待分配；不用它们给主组额外缓存或把C8与W8A8成绩混合。旧4P2D归档仅在明确溯源时按需读，不继承PID/配置/性能。
