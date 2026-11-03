# EP8 L24 Indexer 的 metadata 生产记录：新增线索与长度缺口

本次只读 EP8 已保存数据库，围绕三条已精确识别的第 24 层 Indexer 样本定向寻找其可能的 metadata 上游，未读取张量元素、提示词或权重，也未启动 NPU。查询时验证 DB SHA-256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`。

## 同一采集中存在 metadata 生产任务

`STRING_IDS` 里有 `VllmAscendQuantLightningIndexerV2Metadata`；以其 opType 关联，整个 device 8 的 TASK 表有 20 行。之前 `request-dynamic-metadata-coverage.md` 的“六个固定术语未命中”只针对当时的关键词和查询范围，**不能扩大成该数据库没有 Indexer metadata 相关名称或任务**。这次每个目标 Indexer 前按导出设备时间取最近两条 metadata 任务，各条 globalTaskId 在 TASK 表中唯一，各通过 connectionId 关联 CANN `launch` 与同名 host ACLNN API；所选 host API 的输入形状/类型字段及 callchainId 都为空。

| L24 样本 | 目标耗时 | 最近两条 metadata 的 globalTaskId | metadata 导出输入 shape | 输出 shape | 结束到目标开始的导出坐标间距 |
| --- | ---: | --- | --- | --- | --- |
| 短轮 1 | 32.020 μs | 1801 / 1800 | `[5;;;4;]` / `[5;;;4;4]` | 两条均 `[1024]` | 26.476 / 26.630 ms |
| 长轮 2 | 498.491 μs | 13299 / 13298 | 同上 | 两条均 `[1024]` | 25.234 / 25.379 ms |
| 长轮 3 | 498.170 μs | 14478 / 14477 | 同上 | 两条均 `[1024]` | 25.898 / 26.118 ms |

metadata 的导出 `attrInfo` 为空，所选 host API 也没有输入 shape 或调用栈；记录没有给出这两种输入 shape 各元素的值。目标 Indexer 自身输入 shape 中有 `[5]`、`[4]` 和 `[1024]`，与这些 producer 形状相容；**时间接近及形状相同都不是 tensor/storage 身份或跨流依赖**，无法证明这六条任意一条输出被相应 Indexer 消费、属于同一请求或 replay。上面的时间差不解释为下发延迟或计算等待。

## 源码与官网能补到哪一层

本任务冻结的 `vllm_ascend_quant_lightning_indexer_v2_def.cpp` 依次声明可选 `cu_seqlens_q`、`cu_seqlens_k`、`seqused_q`、`seqused_k`、`cmp_residual_k` 和 metadata 等输入；arch22 内核 `quant_lightning_indexer_v2_kernel_arch22.h:221–312` 在对应布局和指针非空时，从 query 的前缀差以及 key 的长度和 residual 获取长度。此源码与历史二进制未绑定，也未证实 exporter 的输入槽位与源码逐项对应；不能把目标字符串中的 `[5]`、`[4]` 升级为确定的有效长度值或请求数。

昇腾当前 [QuantLightningIndexerV2Metadata 官方源码文档](https://gitcode.com/cann/ops-transformer/tree/master/attention/quant_lightning_indexer_v2_metadata)说明它根据长度信息生成各核计算区间，query 前缀和 key 实际长度是数据元素，输出是 `[1024]` 的 metadata。该页面为开发主线、不是本次 CANN 9.2 本机自定义算子的已证明相同版本；其 A3 头数约束与目标导出物理头数也不完全一致，不能直接拿作该历史调用的运行契约。

因此目前仍不能给出三次真实 query 行数、每请求的 cache/key 有效长度、参与的候选块数量，或判定短长差异原因。最小缺口是将每次 metadata 输入和输出与目标 graph replay 的 buffer 身份/生命周期绑定，再获得对应的 `cu_seqlens_q`、`seqused_k`、residual、candidate 内容或可靠匿名聚合，以及实际二进制和布局版本；CPU 侧 optimistic 长度、物理缓存容量或原始请求 32K 仍不可代替这些数据。本轮不能新增 NPU 采集，上述只记录将来能够区分的证据，不是实施计划。

## 复现

`audit_indexer_ep8_metadata_producers.py` 只读既存 SQLite，校验目标唯一与 DB 摘要，保存 `indexer-ep8-metadata-producers.json`。候选规则是各目标之前结束的最近两条相同 opType 记录，非 producer→consumer 自动匹配；缺值保留 unknown。
