# GLM-5.3-Flash 项目作用域

project_id=`glm53-flash`。本目录是独立项目，默认只读本目录HANDOFF/RESOURCES/PLAN/STATE；本地测试合同与执行规则分别为BENCHMARK_CONTRACT/EXECUTION_CONTRACT。不加载父目录混合状态或另一模型的AGENTS/下一动作；它们不构成本项目指令。显式历史证据仅按需只读。

- 先建立Flash的vLLM文本/PD参考，再完成SGLang Flash混合状态/KDA与1P1D适配，后单图/多模态和性能。
- 只操作本项目白名单节点和自有任务/资源；模型路径、architecture、quantization、tokenizer/模板、runtime库、图/MTP/缓存均在本项目独立冻结。不能仅凭`config.json`相同SHA判定两个量化变体等价。
- GPT6.1 Sol High主控；所有实际资源、编译、部署、服务、模型、压测/profile经本机ZCode DeepSeek执行。主控审核SHA/精确launcher一次、独立验收；不读provider/密钥，不重放旧launcher。
- 使用独立工作目录/分支以及/data/wio/inference-foundry/glm53-flash任务namespace，独立配置/脚本/锁/Meta/provider/结果；namespace尚未由此次文档拆分创建。两个对话不共享可变Current/合同/状态/审批队列，不等待对方模型进度。
- 测试只比较本模型两框架。预热排除、不同独立前缀/四轮新后缀等口径看本项目合同；参数/合同变更在本目录版本化，不要求另一模型同步。错误/超时/缺字段原样保留。
- 性能前核现场实际负载/链路背景争用；物理资源争用只是本次测量条件，不能因此控制另一个对话或拿它的结果拼接本项目指标。
- 公共网站用HTTP/API/Web Search，不用Browser/Chrome/ComputerUse。现场不联网，离线依赖在Mac下载校验后传入。禁止改系统SDK/驱动/BIOS/时钟/整机重启，禁止恢复定时或自行创建/消息其他对话。
- Git只交源码、配置、身份/证据索引，禁止凭证/provider/权重/镜像二进制/原始业务文本。只提交本项目目录，公共索引需要变更时不覆写其他项目。

P→D数据面强制使用**SDMA**，同时适用于本项目vLLM-Ascend与SGLang；不允许自动回退RDMA传输模型状态。按本目录[SDMA合同](PD_TRANSPORT.md)核配置、实际加载路径和真实请求传输证据；TCP控制/握手与PD数据面分开。此项为用户指定要求，尚不是运行验证结论。
