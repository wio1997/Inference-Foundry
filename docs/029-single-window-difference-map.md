# GLM-5.3-Flash：32K 单窗口差异地图与客户端请求分布

**状态：单次诊断与首轮实测的离线描述，不是最终因果地图，也不是新的压测或优化验收。** 本文固定已有归档基线 commit `8a14be830aab38d6ceab7fcc15815308a3a5f1fa`，将正式冷/热客户端测量与单独开启 profiler 的 32K 诊断窗口分开；没有逐 shape、逐 layer 或相同逻辑 step 的跨框架匹配。这里的差异不代表框架本身的因果差异，也不代表可消除的墙钟时间或吞吐收益。

## 一、口径与证据范围

- 模型 GLM-5.3-Flash W8A8、同一对 910B1 P/D 机器，各侧 8 卡；TP8/EP8、MTP3。两框架正式测量为独立服务 epoch，SGLang/vLLM 的 CANN 分别为 9.1/9.2、torch_npu 版本不同；P/D 最大运行请求数 SG 16/16、vLLM 64/32。相同物理机器和客户端输入不消除这些混杂因素。
- 正式结果为 AISBench **32K/64K 各一轮 cold 和一轮 hot**，每阶段 32 个成功请求、客户端并发 C16、每请求 1024 输出。cold/hot 均在已完成预热后，hot 不是必然命中 prefix；这四组不是独立重复或随机 A/B。
- 单独的 **32K C16 诊断 profile** 每框架各 P/D 八 rank，共 32 份 compact。两框架诊断数据解压后各 16 行、整文件 SHA-256 相同；同数据集和 C16 不等于同模型内部 batch、shape、服务进度或步骤。SG 的 P 窗口含 bs=1、8192-token EXTEND 和 draft；D 窗口含 bs=1/query=4 VERIFY。两端均关闭 `record_shapes`；vLLM compact 缺少与 SG 窗口一一对应的 step/shape 证据。
- 原始大 trace 不在这些 compact 内；SG 硬件摘要是截断的 top50，vLLM 通信摘要含名称启发式分类。没有导出的事件**不等于零**；累计设备/AICPU/SDMA/WAIT 可能重叠，不能相加为墙钟，也不能跨未校准主机时钟相减。

来源：[vLLM 16 rank 采集与验收](012-vllm-G4-all16-model-profile.md)、[SG 16 rank 采集与验收](018-matched-SG-all16-model-profile.md)、[原有机制地图及版本限制](020-profile-mechanisms-and-priority-candidates.md)。原始数据和各报告的固定身份由[归档清单](../SHA256SUMS)和[来源记录](../provenance/artifacts.json)核查。

## 二、32K/64K 首轮：客户端配对不等于服务端配对

以每一阶段的 `data_id`、实际输入字符串 SHA-256 在内存中逐条配对，两框架各阶段均 **32/32 输入匹配且请求成功**。每个客户端从自身 SQLite 复算时长，先在各自时钟域内求差，再比较时长；不直接相减跨进程时间戳。

| 输入/阶段 | SG / vLLM 输出吞吐 token/s | SG / vLLM 平均 TTFT 秒 | SG / vLLM 平均 TPOT ms | 配对请求 TTFT 差中位数（范围），秒 |
|---|---:|---:|---:|---:|
| 32K cold | 149.55 / 208.09 | 65.70 / 41.58 | 22.6 / 22.1 | +26.94（+2.64 至 +36.08） |
| 32K hot | 148.18 / 208.36 | 66.00 / 40.70 | 22.7 / 22.7 | +27.94（+2.81 至 +40.07） |
| 64K cold | 71.38 / 101.46 | 156.18 / 106.17 | 23.0 / 24.3 | +59.63（+4.16 至 +70.67） |
| 64K hot | 71.37 / 101.28 | 156.12 / 106.29 | 23.8 / 24.3 | +59.08（+4.38 至 +69.37） |

差为 SG 减 vLLM；四组 TTFT 均为 32/32 配对请求方向为正。首 token 后至最后一次流式事件的配对差中位数依次为 **+1.98、+2.11、-3.09、-1.99 秒**，其方向并非逐请求一致。这些是同输入、两次服务运行之间的客户端时长观察；32 条并发请求不是 32 次独立重复，也不是相同服务端 batch/step。TTFT 包含客户端、排队、路由、P、PD 及 D 首 token 处理，不能等同 P 计算时间；首 token 后区间也包含网络与发送，不能称为纯 D kernel。

记录和五项原始指标：[32K cold](../records/points/GLM-OPT-0001/runs/GLM-RUN-0048/manifest.json)、[32K hot](../records/points/GLM-OPT-0001/runs/GLM-RUN-0049/manifest.json)、[SG 32K cold](../records/points/GLM-OPT-0001/runs/GLM-RUN-0073/manifest.json)、[SG 32K hot](../records/points/GLM-OPT-0001/runs/GLM-RUN-0074/manifest.json)、[vLLM 64K cold](../records/points/GLM-OPT-0001/runs/GLM-RUN-0053/manifest.json)、[vLLM 64K hot](../records/points/GLM-OPT-0001/runs/GLM-RUN-0054/manifest.json)、[SG 64K cold](../records/points/GLM-OPT-0001/runs/GLM-RUN-0081/manifest.json)、[SG 64K hot](../records/points/GLM-OPT-0001/runs/GLM-RUN-0082/manifest.json)。

## 三、32K 诊断窗口：八卡摘要，不是逐 step 因果归因

每项先对每 rank 求 `sum / count`（或保留该 rank 的累计 sum），再在**同一次采集的八个 rank**上求中位数。八张卡不是八次独立试验。下面事件即使名称和计数相同，也尚未验证 shape、target/draft 构成、上下文和逻辑工作单位相同。

| 侧与事件 | SG 观察 | vLLM 观察 | 可支持的范围 |
|---|---:|---:|---|
| P：KDA | 每 rank 272 次，均值的八 rank 中位 4.007 ms/事件 | 每 rank 272 次，3.253 ms/事件 | 两种实现与聚合方向值得做同 shape 核查；不能当作端到端耗时差。 |
| P：同名稀疏索引 score | 每 rank 96 次，12.375 ms/事件 | 每 rank 96 次，15.938 ms/事件 | 存在相反方向观察；不能推断同名即同工作。 |
| D：同名稀疏索引 score | 每 rank 96 次，累计时间的八 rank 中位 27.17 ms | 每 rank 96 次，中位 8.69 ms | 单窗口累计设备事件线索，非请求延迟和关键路径贡献。 |

已有[机制地图](020-profile-mechanisms-and-priority-candidates.md#已确认的实现差异)还记录 SG P 的 DeepEP dispatch/combine 和 GMM、独立激活量化，与 vLLM 的 fused GMM 路径不同。vLLM 的 AllGather 路径是由固定源码和配置**推导**，不是本次逐事件证实每次 MoE 通信的运行时 hook。原生 AllGather 候选仅准备，尚无完整语义和端到端 A/B 结论。

示例原始紧凑报告：[SG P rank0](../evidence/matched_SG_G181/profile_G240/CPU_p/review_g252/rank_0.json)、[SG D rank0](../evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/rank_0.json)、[vLLM P rank0](../evidence/G4_profile_G138/CPU_p/review/rank_0.json)、[vLLM D rank0](../evidence/G4_profile_G138/CPU_d/review/rank_0.json)；复现脚本实际读取并校验 32 份逐 rank 报告，而不是只引用 rank0。

## 四、保留的失败和未验收范围

- SG 128K 的冷/热原始测量保留，但 D 端采集曾指向旧实例；**当前 D 在历史 128K 中无新增编译未验证**，完整协议验收撤回，不能沿用旧通过字段。[更正](025-G316-D-compile-root-correction.md)
- SG 256K 首轮 cold 32 个请求中 18 成功、14 失败，hot 未提交；无完整有效首轮对比值。后续路由健康检查变更在原记录中是候选修复，**长负载收益和恢复不是本文结论**。[失败记录](028-G316-first256-cold-router-failure.md)
- 本地图没有给出排队、P 计算、P→D 传输、D 计算或各算子对 TTFT 差的贡献比例；也不把通信路径和融合差异直接认作慢的根因。需要先拿到同一 rank 内 event/stream/graph 关联，并核对匹配 step、query/context、dtype、专家工作量和时间域；原始 trace 无 shape 时须在资源许可后补采。干预须另做数值、质量、完整 PD 和端到端 A/B。

## 五、离线复现与发布范围

在本仓库根目录运行（Python 标准库，不访问设备或服务、不写数据）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/reproduce_029.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_reproduce_029.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[复现脚本](../scripts/reproduce_029.py)根据仓内原始 xz、SQLite、32 个 compact 和清单重算输入配对、客户端时长差及上表三类 rank 统计；检查每个源文件发布 SHA、JSONL 解压 SHA 和诊断输入整文件 SHA。默认只输出聚合数字及诊断整文件 SHA，不输出输入文本、预测文本、逐请求摘要或时间戳。[测试](../scripts/test_reproduce_029.py)覆盖输入/ID 不一致、重复 ID、缺失事件不等于零以及重复执行一致性。

本次离线交叉核对中，32 个发布 compact 与独立分析使用的原版字节 SHA 全部一致；八个正式阶段共 256 条请求的输入 SHA 与原始见证逐条一致，重算时长也一致。压缩请求归档本身保留原始输入字段：**脚本不输出这些字段，不代表历史归档已经完成数据内容脱敏；公开发布仍需依据原始数据授权单独审核。**本校验也不能从 compact 还原缺失的 shape、逐事件因果或原始完整 trace。
