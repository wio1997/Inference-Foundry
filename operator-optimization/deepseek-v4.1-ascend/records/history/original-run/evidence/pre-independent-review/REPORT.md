# 169 vLLM-Ascend：融合扫描、低活动模块和条件理论界

更新：2026-10-28T09:01:25.803644+00:00。结论范围是已有证据的发现与定位；没有修改模型、框架或 kernel，没有实施融合或调参。本轮仅 CPU 离线分析既有 16 rank trace，GPT 主控，本机 ZCode DeepSeek 执行机械任务。

## 当前发现

已验证归属的 40 个 target 解码层中，第 0 层的 CSV 报告 AIC 周期加权 MAC 活动指标最低。去掉采样边缘后为 **3.6402%**；全采样口径为 3.0315%。该指标是报告字段构成的代理量，**不是整卡利用率、MFU 或相对物理峰值的算力效率**。

第 0 层内部 MC2 的 48 次调用平均 942.266 微秒，其他 39 层对应均值的中位数为 303.045 微秒。现有时间线支持跨 DP 到达不齐、早到 rank 在 MC2 中等待的机制；尚未证明其可消除时间或端到端影响。不能据此认定底层 INT8 计算慢。

## 基线与数据范围

无 profiler 基线保持为 32K 输入、1K 输出、并发 8、实际前缀命中 89.990234375%；两轮聚合 **519.8440965 输出 token/s**。基线与有 profiler 的短样本分开，不用采样耗时替代基线。源版本、镜像、数值语义见 contract.json；原始导出 CSV 哈希见 evidence/TRACE-EXPORT-SHA256.json，原始大 trace 保留在 169 本任务原目录。

全量聚合覆盖 16 rank、481548 行 kernel CSV。通过 mHC 次序、每层 1 MC2/1 SparseFlashMla、压缩 KV 与 index source 角色等独立不变量，从捕获图恢复 4160 次 target decoder 调用；16 rank 全部通过。主排名使用每 rank/model 的 runs1、2、3，共 1920 次调用、每层 48 次，先固定采样规则再比较。全采样与内部样本的最低层身份一致。

| 已测模块 | MAC 报告代理指标 | 平均模块窗口/微秒 | 样本 |
|---|---:|---:|---:|
| 第0层 | 3.6402% | 1419.587 | 48 |
| 第16层 | 7.5786% | 743.945 | 48 |
| 第20层 | 7.5951% | 1027.922 | 48 |
| 第10层 | 7.6970% | 733.662 | 48 |
| 第9层 | 7.8349% | 774.998 | 48 |

分子为 `sum(aic_total_cycles*aic_mac_ratio)`，分母为 `sum(aic_total_cycles)`，仅纳入已归属 kernel；不是百分比简单平均。PMU 字段审计未发现正周期但比例缺失/非法的行。缺周期的任务没有被当成具有已知 PMU 的零活动任务。完整分子/分母、各 rank 结果和全部 40 层排序在 module-analysis.json、captured-layer-attribution/ 中。

覆盖缺口：eager prefill、3 个 eager DSpark block、decoder 外的 Engram CPU lookup/投影门控、root/head/metadata 未形成同口径模块排名。这里只能称“已测 40 个 target decoder 中最低”，不能称整个推理框架最低。模块窗口的跨 rank 平均值不是全局关键路径时间。

## 融合扫描：做了什么，停在哪里

冻结 G0，G1 补齐延后消费的 mHC 系数跨层依赖。12 个源码模板、221 个模板节点，扫描两两、三个、四个相连节点、共享输入分支以及已知复合阶段的拆分重组/跨模块边界，枚举 928 个结构项；第二轮同图同规则新增 0。928 是结构项数量，不是已证明可实现或能提速的融合数量。

对 31 个源码边界检查项复核：27 个有条件候选，3 个已实现的融合（mHC pre、RMSNormCast、SparseFlashMla），1 个 metadata 路径仍未知。候选图在 candidates/，原始检查在 evidence/FUSION-SWEEP.md，覆盖/停止在 fusion-scan.json，重叠关系在 records.json。已实现融合不重复算收益；相互重叠的候选不能把节省量相加。

具体例子 F05：Q 归一化输出 qr 在 8 个 index source 层还供给 indexer，融合 norm+quant 不能直接删除 qr 写出。其余 32 层没有该数值消费者时，qr 的一写一读最多为 5120*M 字节的有条件移除机会。BF16 舍入、scale 归约、两条分支及原有多流重叠必须保留。详见 evidence/QR-consumers-and-fusion-review.md。

当前融合阶段标记 `PAUSED_WITH_GAPS`，指完整 primitive/alias/proposer/metadata 图和语义证明仍有缺口；**不表示整个任务暂停，也不表示已经证明再无可融合项**。不能把有限结构复扫包装成全模型融合穷尽。

## 理想上界与实际对照

先允许理想融合和重叠，再用必要工作/容量约束：`T>=max(F8/P8, F16/P16, D_HBM/B_HBM, D_link/B_link, L_causal,...)`，`U=Q/L`。共享资源只在不重复计数时累计工作；当前同步、Python 转换和所选 tile 的开销不自动进入不可避免下界。详见 evidence/IDEAL-MODEL.md。

官方标称公式与实际设备信息给出条件分母 353.8944 TMAC/s/逻辑设备；9382 适用性、1800 MHz 频率上限及标称公式作为容量上限仍是前提。以下独立 Wq_b 是所有 M 行输出需保留的标准 dense 算法域，M 不等于返回 token 数。

| 对象 | 实测中位数/微秒 | 条件计算下界/微秒 | T/L（非可达提速） |
|---|---:|---:|---:|
| Wq_b_rank0_M3 | 41.919 | 0.355556 | 117.897 |
| Wq_b_rank0_M416 | 156.557 | 49.303704 | 3.175 |
| Wq_b_rank0_M1 | 40.399 | 0.118519 | 340.867 |
| Wq_b_rank0_M2 | 40.439 | 0.237037 | 170.602 |
| Wq_b_rank0_M817 | 242.215 | 96.82963 | 2.501 |
| Wq_b_rank0_M419 | 151.817 | 49.659259 | 3.057 |

对选定 target INT8 必要工作，用 16 设备容量得到的 **116294.745 输出 token/s** 只是极松的条件计算上界：忽略 attention、BF16/FP32、draft/reject、访存、通信和调度。不能把它当“本机实际极限预测”，不能声称基线存在 223 倍可提速空间。

本次已保存 7 个独立数值对象的精确输入与计算器原始输出；40 层模块和 27 个未实现候选也有对照记录，缺失项保留符号/null。HBM/L2/互连容量上限、必要字节和有效路由计数未证，**本机无条件、不可超越的数值吞吐上界仍未知**。没有用最快实测速率代替硬上限。

## 第 0 层为什么低：支持到哪里

MC2 源码已融合 dispatch、GMM1、SwiGLU/量化、GMM2、combine，并有逐 expert/跨核心 flag 与跨 rank readiness 轮询；它不是一个纯 INT8 GEMM。源码见 evidence/MC2-known-dependencies.md。

DP0 记录 400 次 target MC2，DP1 记录 440 次。首次等次数校验主动拒绝配对，失败记录保留；后续保留不等次数并检验相对偏移 -1/0/+1 轮。offset0 的 400 组调用均有 16 rank 公共执行区间，错开一轮均没有；这支持该配对，但没有内部 operation ID 与时钟误差上限，仍作为有条件的时间线关联。

在 3 次两组 DP 都处于内部采样窗口的第 0 层调用中，层入口的开始时间差中位数 1347.468 微秒，MC2 开始时间差中位数 1344.596 微秒，MC2 结束时间差中位数 32.524 微秒。每次都是 DP0 晚进入该层，DP1 早进入；各组层入口到 MC2 的耗时约 377–395 微秒，差异小得多。时间差在层入口前已存在，不能归咎于本层 attention 本身多算了同等时间。

这些证据与早到 rank 在 MC2 等对端一致。当前 TASK 连接到的是重复的捕获定义，104 次第 0 层精确 TASK 查找均成功，却未找到对应的每次 replay CANN 提交连接；不能靠 CPU 时间窗包含关系补造归属。因此，DP0 为什么更晚进入图——host 提交、前序设备队列、其他工作或 profiler 扰动——仍未区分。

小 M/专家负载不均/访存与当前分块策略可同时贡献；未拿到专家有效 token 分布、内部发布时间与实际容量，不能精确分摊。去掉边缘后异常仍在，削弱“纯采样首尾效应”的解释，但不排除 profiler 对调度的扰动。基线端到端可消除时间为 UNKNOWN。

## 复核与恢复

先读 STATE.json、contract.json、profile-summary.json 和本报告；再按索引读取最多 200 行/32 KiB 的局部证据。完整聚合保留，不把整个大 trace 塞入模型。tools/ 保存 CPU 分析脚本，bounds/ 与 calculations/ 保存精确数值复算；calculator 仅检查算术，前提由主控审阅。ZCode 模型验证和执行失败/成功凭据在 zcode/ 与 evidence/ZCODE-EXECUTION-INDEX.json。

本轮分析记录同步到 `/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028`。不操作 172，不修改目标框架/模型，不发送新推理负载。

此前目标工具曾持续返回无原因的 blocked 与旧时间戳，期间离线工作继续执行。最新复查已返回 active，目标模式状态恢复；此前状态异常的原因仍未知。详见 evidence/GOAL-STATE-ANOMALY.json。
