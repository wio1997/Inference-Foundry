# 169 研究阶段交付与恢复入口

> **最新用户指令：任务未完成，转新对话以目标模式继续。** 此处历史“已完成/交付”不代表当前验收；先读 [CONTINUATION-STATUS.md](CONTINUATION-STATUS.md)。

完整目录、路径映射与环境信息见 [RESEARCH-INDEX.md](RESEARCH-INDEX.md)，结构化路径与哈希见 [RESEARCH-CATALOG.json](RESEARCH-CATALOG.json)。环境已于 2026-10-30 18:12（Asia/Shanghai）只读复核，实际启动脚本/容器挂载见 [环境快照](ENVIRONMENT-SNAPSHOT-20261030.json)。

本轮目标是用户 2026-10-30 的六步研究报告，不是修改模型/框架，也不是其他机器的 W4A8/910B3 项目。继续时先读本文件、STATE.json 和 publication-draft/README.md 正文，历史折叠附录按需查阅。

报告已发布到 Inference-Foundry 的 analysis/ascend-910c-performance 分支：commit `9e2870bb9982d83377b180f9523adadf8e5343bd`。12 个文件在 Git tree 中与本地 blob SHA 一致，branch ref 已核对。publication-record.json 保存完整校验记录。

- 场景：DeepSeek V4.1 Flash W8A8，169/910C，TP8×DP2，8 请求×32768 输入/1024 输出，命中 235904/262144，0 抢占。
- 无 profiler 客户端参考：16.004567 s，聚合 511.854 输出 token/s。历史 AISBench 519.84 口径不同。
- 官方 run：official-20261030T084726Z-cb252d1d，两个窗口×16 rank 已导出，全部模型/流按类型和完整 shape 分析，原始文件仍在服务器 main 目录。
- CPU run：framework-20261030T091949Z-3d9bcf87，ZCode 执行一次，两个真实 EngineCore 外部栈与 proc 计数采样成功。已有 async batch queue，回复队列路径值得查，但样本比例不是可删时间。
- 结束状态：原 baseline 容器运行且健康，两个引擎 idle，NPU 进程均属本项目；没有在跑的本次设备实验。不操作 172。

优先机会：初始大行数 MoE 内部交接/有效专家工作、Indexer token-K 特化、单组输出投影接口、Q_b head 分片前移/权重复用、已有 batch queue 的回复与下一批准备边界。mHC/KV/Engram 等候选见报告。现有融合、分组流水、前缀缓存、图和异步机制不能重复计作新收益。

未知仍明确保留：同次有效专家 R/A、真实 HBM 字节、完整端到端通信暴露、逐任务 phase/身份、CPU-NPU 同次因果对齐、真实权重上的候选精度/收益、严格数值全机吞吐上界。报告已完成不意味着这些问题已解决，也不要求继续无限取证。

四个历史 Task 与其 Loop 原样保留，index/resume 标明本研究阶段已交付。不要将旧 E0、旧 active_loop 或旧 next_action 当作本轮仍在运行的证据；以当前目标、真实进程与新报告为准。下一阶段若开发具体候选，按实际缺口冻结一个小范围试验，不重新开始整张图枚举或合成 dump。

复算：publication-draft 下 verify_summary.py、verify_official_capture.py、verify_official_operators.py、verify_framework_cpu.py 均已通过。它们核对公开摘要算术，不能独立重采私有权重/输入或证明硬件容量。简单启动/导出/采样已使用本机 ZCode DeepSeek；超时的机械提取未记成功。

文件同步回执位于 diagnostic-20261029/phase-sync-receipt.json，逐文件 SHA256 核对；只写本人容器映射的 `/data/tiankuan/wio/main` 任务目录。
