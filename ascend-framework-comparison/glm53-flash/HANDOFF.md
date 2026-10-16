# GLM-5.3-Flash 当前交接

本项目独立于另一个模型的对话。此前混合总入口已停用；默认只读取本目录。节点与预期模型身份以[本项目RESOURCES](RESOURCES.json)为准，具体源/目标证据在[ASSETS](ASSETS.md)。本项目资源准备由本对话完成，接手者可能已在现场推进，启动前核实际进程/锁，不把本文资源记录当实时服务状态。

先建立Flash的vLLM文本/PD参考，再完成SGLang Flash混合状态/KDA与1P1D适配，后单图/多模态和性能。

权重为`/data/wio/models/GLM-5.3-Flash-w8a8`，glm5_next / Glm5NextForConditionalGeneration。候选Docker实际ID均已独立验证，尚无本资源任务产出的新模型服务/性能基线。源码快照与旧资源批次目录只作证据，不是要恢复的服务。当前新运行namespace`/data/wio/inference-foundry/glm53-flash`为计划，未在这次文档拆分中创建。

GPT6.1 Sol High主控、本机ZCode DeepSeek执行真实动作；详见[本项目执行规则](EXECUTION_CONTRACT.md)。不恢复定时，不自动创建/消息另一对话。只改本项目的配置、脚本、数据、STATE、结果和run；不需要另一模型跑通、同意或改合同。

下一步：读取本项目PLAN，核独占与image/模型身份，经本机ZCode冻结脚本开始本模型正确性门控；完整通过后用本项目BENCHMARK_CONTRACT建立基线。预热不计吞吐，不共享长前缀，失败全保留，实际缓存与运行/排队以证据为准。

Flash vLLM14P/15D，SGLang46P/47D，各1P1D。先模型加载、KDA/conv/recurrent与mHC/量化语义，再混合状态PD、图/MTP和缓存；多模态有本目录独立方案。不要从原GLM53的MLA/MemCache问题、层数、73K命中常数或2P2D规模继承推理配置。

P→D数据面强制使用**SDMA**，同时适用于本项目vLLM-Ascend与SGLang；不允许自动回退RDMA传输模型状态。按本目录[SDMA合同](PD_TRANSPORT.md)核配置、实际加载路径和真实请求传输证据；TCP控制/握手与PD数据面分开。此项为用户指定要求，尚不是运行验证结论。
