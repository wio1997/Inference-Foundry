# GLM 32K D 侧 rank0：两个独立局部时间轴

**状态：既有诊断 trace 的独立局部事件样本，非同一请求/step、整模型或性能排名。** SGLang 和 vLLM 各选一个 D rank0 100 ms 窗口并分别归零；事件可跨 stream 重叠，599 与 569 的可见条数不能用来给框架排名，也不能把累加时长当作墙钟损失。

## 1. 来源、身份与窗口规则

两份来源是 GLM-5.3-Flash W8A8 的 32K 诊断采集，客户端配置 C16；不属于正式性能分数。SGLang 是 G181/G240 冻结采集导出的 D rank0 trace；vLLM 是 G4/G138 冻结原始采集的 G148 **派生分析副本**，不是原始 profiler 二进制。[历史 D 逐事件与来源记录](030-decode-rank-event-distributions.md)

| 样本 | 源 trace SHA-256 | 字节数 | 首个白名单事件相对偏移 | 选窗内事件 | 显示坐标轴 |
|---|---|---:|---:|---:|---:|
| SGLang | `6750d2b2b57c13fba507a4c1af9776e30d4f93c51c9fe44a12f49150c23837be` | 49432057 | 542768.5 μs | 599 | 100.645 ms |
| vLLM | `3d1b92c234cd2ce1879caeed340baf0272ac70a87523d8fd224d762bcd014a43` | 32280185 | 66513.25 μs | 569 | 100.000 ms |

源端提取器在各自 trace 的 rank0 设备白名单事件开始时间中，选取**按源时间对齐的 100 ms 桶**中事件数最多者，同数取最早；以该桶内首个选中事件独立归零。所选事件均按开始时间落入桶内，若末事件结束超过 100 ms，显示轴随之延伸：SG 为 100.645 ms；vLLM 为 100 ms。两个样本都少于 1000 条，没有下采样。不能将两轴相同坐标当作同一 step，也无法从局部样本重新证明原 trace 的全局最密桶。

源端提取记录为两遍只读读取；每遍验证 trace SHA、文件大小和 mtime，且两遍记录数及设备元数据一致。SG 的 trace SHA、大小、mtime 与 G240 冻结索引匹配；vLLM 的派生 trace SHA 与 G148 rank 收据匹配，另有 **13 个 frozen raw 组件**逐项与已有索引匹配。组件核验与派生 trace 的关联是已有来源链，**不代表此次重新执行 SDK 转换**。本轮只消费已有本地样本，并未重新访问设备或读取原 trace。

- [SGLang 事件与来源摘要](../evidence/d_timeline_037/sglang_rank0.json) · [vLLM 事件与来源摘要](../evidence/d_timeline_037/vllm_rank0.json)
- [SGLang D rank0 收据](../evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/rank_0.json) · [vLLM D rank0 收据](../evidence/G4_profile_G138/CPU_d/review/rank_0.json) · [13 组件的既有脱敏收据](../evidence/trace_030/vllm_rank0.json)
- 本地原局部样本 JSON SHA `468607f30a4d6a8723478ff0ed6be6bd20e70b3ea84e2f81aefd25c8256fb455`；原静态 HTML SHA `526d192f3dc7cd7c25c06f7bff9d86c658aad32fb0edd25c16ff6875109fe154`。本仓不复制原 HTML、带私有路径的提取脚本或大 trace。

## 2. 窗口内事件与精度

| 类别 | SGLang 选中事件 | vLLM 选中事件 | 限定 |
|---|---:|---:|---|
| 普通 GroupedMatmul WeightNz | 126 | 101 | 未按相同 shape、专家输入、step 或工作量配对。 |
| 融合 GroupedMatmul+SwiGLU+quant | 本窗口未识别 | 101 | 不代表整个运行不存在，不能把事件次数之差当收益。 |
| QuantBatchMatmulV3 | 187 | 309 | 名称相同不代表输入 shape、图或 token 相同。 |
| lightning indexer / SparseFlashAttention | 16 / 18 | 27 / 31 | 单次局部计数，不是同请求路径。 |
| MoE LowLatency dispatch / combine | 各 63；相应 AICPU 各 63 | 本窗口未识别 | 设备与 AICPU 事件可能重叠，不是可直接求和的通信墙钟。 |

“未识别”只指该白名单局部选窗，不等于完整 trace 为零。stream 编号只在各自采集中有意义，不可跨框架视为同一执行流。源端筛选真实 `duration > 0`，归档时独立将相对起点与 duration 四舍五入到 **0.001 ms**；当前两个样本没有显示为 0 的 duration，但若未来正 duration 被舍入为 0，不能当作真实零时长。该选中数据的 shape 字段均为空，不能补猜形状；这不意味着同一次采集的其他设备 CSV 都缺 shape。

## 3. 离线验证与查看

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/timeline_037.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_timeline_037.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
# 可选：输出独立、自包含的双面板静态 HTML
PYTHONDONTWRITEBYTECODE=1 python3 scripts/timeline_037.py --html > timeline_037.html
```

[校验/渲染脚本](../scripts/timeline_037.py)只读两份白名单 JSON 和既有 rank/组件收据，验证来源 SHA、rank0、13 项组件、100 ms 桶、1000 条上限、各事件可视轴范围、名称/数字 stream、数量和精度标记；`--html` 将两个独立 SVG 输出到 stdout，不读取 trace 或连接设备。合成和实际归档[回归测试](../scripts/test_timeline_037.py)核验 2 轴、22 刻度、1168 事件条、未截断的 SG 100.645 ms 轴，以及非法字段、rank、大小、时间和来源篡改拒绝。

归档数据只含事件类别、数字 stream、相对起点和毫秒时长，及固定元数据；没有设备 PID、绝对时间、私有路径、IP 或请求文本。本次脱敏副本已与源局部样本的 1168 条事件逐项比对，但本仓不能据此重建完整原 trace、请求级时间线、跨框架同 shape/step 或端到端收益。后续需要按请求/forward、shape、依赖和时间域另行匹配，才能推进机制归因。
