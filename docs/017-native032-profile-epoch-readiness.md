# 原生0.32独立profiling实例：就绪与采集边界

原生epoch630的[四档首次矩阵](016-native032-four-tier-first-matrix.md)已冻结归档，随后按现有控制器机制受控退出。新epoch727使用相同arm64镜像、95文件权重及数学配置，TP8/EP8/DP1/PCP1、DSPARK5、P eager、D FULL_DECODE_ONLY、P/D .88/.80、chunk8192/1024。新实例用于诊断，不覆盖或重跑旧首次正式批次。

[父级现场验收](../evidence/profile_epoch_G727/parent_selected_readiness.json)核对两端实际容器、控制器、八卡owner及锁FD，原生113项源码与额外5项运行契约SHA一致。D八rank日志均完成图捕获，target13档、draft11档；捕获仍不等于实际图重放profile。两项自然文本答案及1024输出SSE通过，服务usage与独立P generation增1/D增1024一致。这是新实例的选定运行门槛，不替代原epoch的1319题质量证据，也不构成全模型或跨框架性能验收。

新实例启用原生profiler API，并通过原生worker-cls扩展点添加[CPU调度注释类](../code/profile_epoch_G727/logical_profile_worker_g746.py)。它仅在profiler运行时读取CPU调度元数据，以record_function记录请求ID、scheduled query、computed/output计数和draft proposals；原生execute_model只调用一次，返回值不改。不读取tensor、不同步设备、不更改模型或kernel。scheduled query不是物理graph padding，draft proposals也不是accepted tokens；两者仍需同窗口trace与计数器确认。该外部instrumentation必须披露，不能把profiling时延当正式baseline吞吐。

已冻结独立16条32K输入和两框架配置，[manifest](../evidence/profile_epoch_G727/fresh_inputs_manifest.json)记录完整原始SHA、逐question/token-ID SHA及16个不同前缀，CPU tokenizer检查它们与原448个预留前缀不冲突。该数据用于双方匹配诊断，准备阶段未向模型发送。旧warmup1诊断输入计划未提交并已被此独立输入替代。

预热继续使用五轮独立warm16/C16/32768/1024。首轮16/16原始请求计数与C16已[父级复核](../evidence/profile_epoch_G727/parent_warmup1_raw.json)，第2/3轮正在执行。最后两轮duration/TTFT稳定、D编译artifact与P独立并集检查完成前不采集profile。[专用预热副本差异](../code/profile_epoch_G727/warmup_pipeline/source_derivation.json)强制epoch727且warmup_only，允许已归档formal数据仅参与只读preflight/resume身份检查；在正式claim之前返回，保留全局首次ledger和原始runner不变。此副本不能被切换成正式压测器。

[当前状态](../metadata/profile_epoch_G727.json)明确保持all-rank profile/replay未验收。尚需实际采集、核对双方P/D每rank设备活动、逻辑工作量、padding、draft接受率、编译与缓存变化，再形成算子/调度/通信/长尾/无效计算地图；高潜力优化与端到端收益仍待实测。大型trace继续留受控存储，公开记录仅包含索引和SHA。
