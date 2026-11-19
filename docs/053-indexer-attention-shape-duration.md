# P indexer / SparseAttention：记录 shape × dtype × format × 耗时

## 目的与边界

在两个独立、非计分的32K P rank0诊断窗口中，基于每条设备CSV事件，将完整记录的输入/输出shape、dtype和format作为**联合签名**，在各组内计算时长分布（p50按线性插值）。进一步只在SG同源trace中，经[已核0/1/多候选链](052-p-heavy-operators-per-step.md)将设备事件与CPU enqueue及标记关联。没有跨框架、跨请求、跨step配对。shape相同也不代表相同tensor值、stride、有效行、mask、上下文工作量或实现；某槽为空、`NULL`或`DT_UNDEFINED`应视为未知，不补零。

## Indexer：同源逐事件的形状与标记关系

SG `lightning_indexer_score_kernel` rank0 96事件分为以下4个联合签名，每组24次。完整字段及输出dtype/format见[统计证据](../evidence/p_indexer_053/joint.json)：

|设备CSV记录的首个输入shape|第二输入shape|末输入shape|n|设备事件p50 (ms)|
|---|---|---|---:|---:|
|8192,128|1,2048,1,128|1,64|24|3.082|
|8192,128|1,4096,1,128|1,128|24|10.834|
|8192,128|1,6144,1,128|1,192|24|13.925|
|8192,128|1,8192,1,128|1,256|24|21.676|

本次**直接逐事件**通过唯一同源候选链发现：16个SG CPU step标记各只有一个indexer联合签名；记录 `c_sk` 为8192、16384、24576、32768时，该签名的第二输入第二维分别是2048、4096、6144、8192。它是在本次记录内的关联，不是将以往shape边际频数与CPU标记频数硬配。每个EXTEND标记对应11次、每个DRAFT标记对应1次；CPU提交在标记区间内不证明设备完成在区间内，更未证层/请求。

vLLM相同事件名共96次、14个联合签名；例如输入 `8192,128;2,5536,1,128;2;2;2,2060` 的11次p50 12.075ms，输入 `8192,128;3,8191,1,128;3;3;3,2060` 的11次p50 20.812ms。SG和vLLM完整记录输入shape字面交集为0：只凭第一输入均为 `8192,128` 不能认为计算同量，也不能把两组p50相减当框架收益。vLLM没有SG风格的step标记候选，不能以SG `c_sk` 给其强制分组。

## SparseFlashAttention：形状不同且未知槽位要保留

SG 96次分为4个联合签名、每组24次，p50约24.177–24.388ms。vLLM 112次分为9个联合签名，既有首维8192的84次，也有首维7680的12次及首维1/2/3的16次；例如第一个输入 `8192,8,512`、第二输入 `8305,128,1,512` 的若干组p50约21.14–21.19ms。SG相同第一输入的第二输入是 `2289,128,1,512`；vLLM字段尾部含 `;;`，记录dtype含 `DT_UNDEFINED`、format含 `NULL`，这些保留为未知。

**不是公平性能结论**：第二输入、其他shape及字段不同，输入有效内容/稀疏mask未知，不能从相同第一输入或这些条件均值确定SGLang慢、慢多少、是否可优化。此处也不把事件时长和当P墙钟或TTFT份额。

## 可复核材料

[联合脱敏统计](../evidence/p_indexer_053/joint.json) SHA `1842ca83d8878351edfd270dce940ca80684b2733bf67ec6e57cd0ba44ce03f5`；与[上一轮stage统计](../evidence/p_heavy_052/step_detail.json)的源SHA、设备候选分母和CPU标记逐项复核一致。SG G240是索引冻结源，vLLM G148为派生分析副本，不是历史冻结原始CSV。分析时两低优先级只读worker、空闲和资源门控、不发模型请求；所有结果只供局部诊断。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_indexer_053.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[脚本](../scripts/verify_p_indexer_053.py)复算来源SHA、分母、先前CPU marker一致性、SG逐标记形状关系、两框架记录字段交集；不认证运行期有效token或端到端根因。
