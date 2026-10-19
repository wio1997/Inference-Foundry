# MTP 的 SFA／Indexer 同步元数据复用

这条研究检查两个消费者是否重复生成相同的 RoPE 查表结果，与已完成的连续表布局优化分开。此前布局收益 C4 +14.71% 属于独立配对结果；修复后的双表 lookup 片段约 0.05ms，不能把调用次数相乘当服务收益。

精确镜像源码中，MTP steps1–4先完成 primary SFA 元数据，再调用 `_build_cache_only_group_next_step_attn_metadata` 构造 Indexer。该回调向 Indexer 传同一个 common metadata，不推进 positions。SFA 和 Indexer 各自调用同一原生 getter；Indexer 随后把结果复制到原有独立 per-step 缓冲。

原型仅在这个回调作用域内把 SFA 的已完成 cos/sin 交给原 Indexer 复制，回调结束即释放选择状态，无跨步 memo。运行时要求 MTP1–4、单个 exact Indexer builder 和 exact SFA builder、int64 同位置地址及 shape/stride。原生表与已接受连续表的所有权保持；不改 target、capture、图指针、原输出事件、InputBatch 或 KV 退休，不新增算子或设备拷贝。源码 pin SFA `9b94f67f…`、Indexer `1f345baf…`、proposer `458fb50e…`。

13项 CPU 检查覆盖回退、异常恢复、同地址位置值改变、前一步独立输出存活及作用域外原生行为。immutable build `202fc1d3d080e8f199d24f06e572cf7529d531e904978cd4271355069b9b9aa7` 有5个 overlay，原 NPU AST 前缀保持，未带提前发布 schema。精确快照并空闲停止旧自有 D 后仅替换 D，保留 P 和官方 proxy，模型、精度、参数及 SDMA 路径保持。

普通功能16条、四并发功能8条数值通过。全部16 rank 共实际复用18752次，逐份与原 getter 及 Indexer 最终副本逐位相同，原 Indexer 存储与 SFA 独立。功能比较会同步，随后关闭 observer，正常异步路径quality52数值全部通过，但完整64仅18/26相同，尚需当前实例native/adaptive/native有限控制，不归因无害舍入。

2026-10-19 15:10 CST，cached16远端执行24.202372s并以rc0结束，下载阶段SSH断开，60011入口TCP主动拒绝；完整metrics待SHA核验取回，尚不记cached通过。没有启动质量控制或正式测量，也没有重复缓存请求。恢复脚本仅拉8份已完成原件；恢复后先核精确实例与有限时长，再推进控制。当前没有本原型服务收益或一般质量结论。

15:29 CST更新：同入口已重连并复核同集群。缓存原8份文件完整取回并核验原SHA，16条数值全部通过，无设备请求重跑。旧P及proxy随后按原有限watchdog自然exit124/OOMFalse；只替换P，D保持同一实例及构建。新旧P immutable源码、图与参数相同，新的实例/engine_id/timer单独绑定；新P Ready后须先8条四并发PD及新增all16比对，再做原8组差异的有限控制。workload和联合门槛保持，未开始本原型正式测量。

证据位于 `evidence/20261018-p-first-seed-device-v2/recovery-v2/`：`rope-step-reuse-source-dependency-audit-v1.json`、`rope-step-reuse-cpu-check-v1.json`、`rope-step-reuse-build-check-v1.json`、`rope-step-reuse-dependency-reviewed-v1.json`。正式联合门槛已在新模型测试前冻结为 `rope-step-reuse-formal-contract-v1.json`，同负载 C1 24条与 C4 ABBA×3 384条，不删除慢块、不把不同窗口比例相乘。
