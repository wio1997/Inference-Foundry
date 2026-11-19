# GLM 32K D：同物理 shape / dtype 的事件分布

**这是原有单次非计分诊断窗口的增量分析，不是相同请求或单变量 A/B。** 固定精确 kernel 名、输入/输出物理 shape、输入/输出 dtype，按同编号 rank 对比 SGLang 和 vLLM 各自的事件分布；两个框架属于不同服务 epoch，实际请求、逻辑 step、tensor 值、有效 token 和 expert group-list 均未匹配，不能由表中差值推出端到端瓶颈或优化收益。

## 覆盖与验证

在[全 32 rank kernel 聚合](039-all-rank-kernel-distributions.md)中，D 侧每 rank 两框架有 77 个精确同名 kernel。其中每 rank 的 9 个 kernel 有至少一个完整输入/输出物理 shape 与 dtype 字符串相同的签名，共 15 个签名；排除了未知 shape/type、`DT_UNDEFINED` 和非类型数字。8 rank 共 120 对物理签名，不是 120 个独立算子或重复实验。普通 GroupedMatmul、indexer 和 SparseFlashAttention 未进入这个严格共享签名集合，**不能称为已按同 shape 配对**。

为取得每个签名的耗时，随后只读 D 端现有 16 份设备 CSV，16 个 rank worker 分别解析；逐源 SHA 与此前收据一致，16×15=240 个框架/rank/签名的实际事件数与已有聚合一致。SGLang 来源匹配 G240 冻结 CSV 索引；vLLM G148 为派生 CSV，SHA 与先前双读收据一致，没有历史冻结 CSV SHA。本次无模型推理请求；I/O PSI 不可用，不能说已验证 I/O 无干扰。

[事件分布和来源摘要](../evidence/d_shape_040/matched_distributions.json)记录每签名次数、sum、mean、p50/p90/p99、min/max 和 stream 分布；源 SHA `3e910b859e4b97176b65553450df69e8721fb375f2f0bfa00acda5894c048347`。`sum` 可与其他 stream 重叠，不可当墙钟。

## 两个例子：有限观察，不是根因

|精确 kernel / 输入物理 shape|统计方式|SGLang|vLLM|限制|
|---|---|---:|---:|---|
|MoeGatingTopK / `4,288;288`|rank0 单事件 p50|5.220µs（8次）|4.680µs（360次）|样本量、请求和 step 不同，不能推断该差值导致 TTFT 差距。|
|QuantBatchMatmulV3 / `4,2048;128,128,16,32;4096;4`|先各 rank 算 p50，再对 8 rank 的 p50 求中位数|约 15.739µs（各 rank 8次）|约 17.180µs（各 rank 112次）|同物理维度不保证同数据和工作单位；8张卡不是8次独立试验。|

同一签名的样本数可相差数倍甚至更多；这些短 kernel 的观测结果不能外推到 32K/64K 的 TTFT 差距。既有 P/D 概览仍显示首 token 前等待较长、D 长上下文准入是待证伪线索，但要判定成因仍需 request/step 关联、可分配/可驱逐 KV、负载和时钟域证据。

## 离线核验

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_d_shapes_040.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[核验脚本](../scripts/verify_d_shapes_040.py)检查固定源 SHA、16 rank、120 对签名、样本数、非负时长/分位数顺序、参考事件及常见私有路径和地址；它不读取原始 CSV、不重跑设备实验，也不能证明两框架执行语义一致。
