# GLM-5.3-Flash：D 侧 rank0/1 单事件分布

**状态：既有 32K 诊断窗口的描述性证据，非同 shape/step 的对照、非正式吞吐测量、非根因或优化收益。** 这里是对 [029 地图](029-single-window-difference-map.md)的 D 侧补充，不能把各事件累计时间相减作为端到端损失。

## 1. 来源、校验与范围

诊断为两框架分别采集的 32K、客户端 C16 输入窗口；SG 为 G181/G240，vLLM 为 G4/G138、G148 分析副本。相同客户端输入不代表服务端相同 batch、query/context、专家 token、target/draft 组成或相同步骤。SG 的 CPU 标签包含 `VERIFY bs=1 g_sq=4`；vLLM 这次未识别到同名标签。两边 `record_shapes=false`，不能确认逐事件 shape、调度、通信或流依赖匹配。

| 框架/设备 rank | 所分析文件的性质 | 字节数 | trace SHA-256 |
|---|---|---:|---|
| SG 0 | 原采集后导出的 `trace_view.json`，不是原始设备二进制 | 49,432,057 | `6750d2b2b57c13fba507a4c1af9776e30d4f93c51c9fe44a12f49150c23837be` |
| SG 1 | 同上，rank1 | 48,945,698 | `3665934e4a7bf82271e0743d515528ac460c3c6f67d0043dfe6a091fd9deca01` |
| vLLM 0 | G148 `analysis_copies` 的**派生** trace，不是 frozen_raw 二进制 | 32,280,185 | `3d1b92c234cd2ce1879caeed340baf0272ac70a87523d8fd224d762bcd014a43` |
| vLLM 1 | 同上，rank1 | 32,282,051 | `bf0801a6a38733ea39628a97e9a6b01ba8de568e144d5a59d8dd1bac9f6ccee3` |

源端原分析对四个 trace 单 rank 流式只读计算 SHA，并记录读取前后大小和 mtime 不变、与先前 rank 收据匹配；**这些是源端既有观察，不是本文重新读取远端的声明**。vLLM 源端还分别核验 rank0 的 13 个、rank1 的 12 个 frozen_raw 组件哈希。组件 SHA/字节数及源端结果保存在下列脱敏摘要，且与仓内 rank 收据逐项比对；派生 trace 与 frozen_raw 的转换关系仍以既有分析清单为来源，本文没有重跑 SDK 或独立验证转换过程。SG 没有声称做了同样的 13/12 组件核验。

- [SG rank0 摘要](../evidence/trace_030/sg_rank0.json) · [SG rank1 摘要](../evidence/trace_030/sg_rank1.json)；[SG rank0 既有收据](../evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/rank_0.json) · [rank1 收据](../evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/rank_1.json)。
- [vLLM rank0 摘要](../evidence/trace_030/vllm_rank0.json) · [vLLM rank1 摘要](../evidence/trace_030/vllm_rank1.json)；[vLLM rank0 既有收据](../evidence/G4_profile_G138/CPU_d/review/rank_0.json) · [rank1 收据](../evidence/G4_profile_G138/CPU_d/review/rank_1.json) · [G148 分析清单](../evidence/G4_profile_G138/CPU_d/analysis_manifest.json)。

## 2. 单个设备事件 duration 分布

时间单位 **微秒**；p50/p90/p99 按该事件集合逐事件 duration 线性插值，不是跨 rank 百分位，也不是请求尾延迟。累计列为窗口内选定事件 duration 之和，可能跨流重叠，**不是墙钟**。rank0/1 来自同一次采集，不是独立重复。

| 引擎/rank | 精确事件类别 | 次数 | p50 | p90 | p99 | 最大 | 累计 ms（非墙钟） |
|---|---|---:|---:|---:|---:|---:|---:|
| SG 0 | GroupedMatmul WeightNz | 720 | 81.692 | 142.809 | 163.483 | 180.404 | 67.817 |
| SG 1 | GroupedMatmul WeightNz | 720 | 33.391 | 58.367 | 78.895 | 94.962 | 25.640 |
| SG 0 | lightning indexer score | 96 | 299.076 | 299.776 | 300.193 | 300.706 | 28.025 |
| SG 1 | lightning indexer score | 96 | 265.585 | 266.645 | 267.177 | 267.406 | 25.128 |
| vLLM 0 | GroupedMatmul WeightNz | 360 | 34.881 | 42.423 | 50.217 | 62.761 | 12.304 |
| vLLM 1 | GroupedMatmul WeightNz | 360 | 35.490 | 42.383 | 48.744 | 51.061 | 12.921 |
| vLLM 0 | GroupedMatmulSwigluQuant | 360 | 55.261 | 79.820 | 100.891 | 106.762 | 19.910 |
| vLLM 1 | GroupedMatmulSwigluQuant | 360 | 55.111 | 80.470 | 92.769 | 96.862 | 20.484 |
| vLLM 0 | lightning indexer score | 96 | 95.642 | 96.232 | 96.746 | 96.822 | 8.837 |
| vLLM 1 | lightning indexer score | 96 | 91.022 | 99.232 | 100.942 | 100.942 | 8.561 |

事件全名和分布见四份摘要。SG GMM 的 rank0 与 rank1 均 720 次，但前者单事件 p50 约为后者的 2.45 倍，是同一次采集中的**排查线索**；不同 rank 的 shape、专家输入和设备/通信状态未匹配，不能据此断言算子实现、卡故障或专家负载失衡。SG 的单事件 indexer p50 在两卡也不相同；跨框架同名 indexer 也仅是线索。vLLM 普通 GMM 与融合 GMM 是不同类别，不能把两者累计值直接加总后与 SG 相减。

SG GMM 在 rank0 的一个 stream 有 672 次、p50 99.222 微秒，rank1 的一个 stream 有 672 次、p50 31.081 微秒；其他 stream 仍有事件，**stream 编号不能假定跨 rank 同义**。rank0 的 GMM 事件有 334 次位于 [100,180) 微秒、2 次位于 [180,200) 微秒；rank1 无超过 100 微秒的该类事件。这是所选窗口中单事件分布，不证明跨时间可重复，也不是 P90/P99 的端到端请求时延。

SG 两个 rank 各识别到 8 个 `cpu_op` step 标签；按开始时间落在这些 CPU 标签区间内的所选设备事件很少。该时间重叠检查**不是** CPU/device 依赖或图 replay 的关联证明。所选事件的 union/span 同样不是模型完整 step 或关键路径；大绝对时间戳转浮点可能引入约 1–2 微秒误差。

## 3. 复现及待补证据

[源端脱敏摘要](../evidence/trace_030/sg_rank0.json)保存了原聚合文件 SHA、源端只读核验的 trace SHA 和选定事件统计；原聚合及大 trace 未收入本增量。本次仅从四份源端聚合构建发布摘要、比对仓内 rank 收据和组件 SHA，**未将实际 trace 带入本地，也未重跑其事件分位数**。因此在只有本仓库的环境下，可以验证已存摘要的一致性，但不能独立重算真实 trace 的 p50/p90/p99。

若后续安全取得对应 trace 的只读副本，可按调用方自行提供的路径和预期 SHA 运行 [通用脚本](../scripts/analyze_trace_030.py)：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/analyze_trace_030.py \
  --trace "$TRACE" --sha256 "$TRACE_SHA256" --engine sg --rank 0
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_analyze_trace_030.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

脚本仅接受小于 50,000,000 字节的顶层 JSON 数组，流式计算自身读入文件的 SHA、校验设备 rank 和读取前后文件状态，只输出白名单设备事件的聚合；不连接机器、不发请求、不输出 trace 路径、任意事件名、请求文本或绝对时间戳。外部所给 SHA 的真实性和 trace 的原始/派生身份仍须另行核验；脚本不会自动验明 vLLM frozen_raw 组件或重新执行 SDK 转换。回归测试使用内存合成 trace 和仓内紧凑报告，不构成对远端真实 trace 的重跑。

下一步应按相同 forward、shape、专家 token 和图 replay 关联验证 SG 两 rank 的 GMM，另核对 indexer 的 query/cache 长度及通信/设备争用；确实存在 shape 和关联数据缺口时再安排有界补采。正式优化需独立数值、PD 语义、质量和端到端对照；不得将本节 duration 或 union 相加成吞吐收益。
