# 新目标下的实质补充：完整迭代中的主模型、输入准备与设备间隔

研究对象：169 / Ascend 910C / DeepSeek V4.1 Flash W8A8 / vLLM-Ascend，32K 输入、1K 输出、并发 8、约 90% 前缀命中。日期：2026-10-30。本文记录新目标 active 期间的追加研究；历史报告发布不作为本次验收。最终整合见[主报告](../publication-draft/RESEARCH-REPORT.md)，交付核对见[六步审计](REQUIREMENTS-AUDIT.md)。

## 这次分析改变了什么

旧报告详细讨论了主模型内部算子，但没有充分解释主模型执行之间的时间。对**同一次官方采集**重新分析后，后续窗口的主模型相容主流段跨度中位为 **37.441 ms**，相邻段之间的间隔中位为 **35.740 ms**。二者均先取每 rank 中位，再取 16 rank 中位，不是全请求时延，也不相加成一个实际样本。

同次 worker trace 中，rank 0 / 8 的 `prepare input` 主机范围在这些间隔内的覆盖中位分别为 **34.006 / 31.485 ms**。这使 worker 输入准备成为必须与 MoE、Indexer、投影一起优先研究的方向。仅依据 EngineCore 回复队列栈，不能完整描述框架侧瓶颈。

**可测事实与因果边界：**`prepare input` 是主机包络，含状态更新、输入准备、元数据构造和同步。它与上一段设备执行有重叠；其完整时长、没有子 scope 覆盖的时间，以及间隔中没有 kernel 记录的时间，都不能直接称为可删除开销。

## 同次数据与边界识别

- 原始 run：`official-20261030T084726Z-cb252d1d`；32 份导出 CSV（两窗 × 16 rank）。新读取的 32 个 `kernel_details.csv` SHA256 全部与原导出摘要一致。
- 不固定历史 model ID 或 stream ID：逐 `(Model ID, Stream ID)`，按开始时间识别重复首 task key，再校验整个 `(Task ID, Type, Input Shapes, Output Shapes)` 序列。
- 找到每次含 **1318 个唯一 task ID、40 个主模型 MoE 形状相容任务**的主流序列。初始窗每 rank 5 次、后续窗每 rank 12 次；各 rank/窗完整签名分别一致。实际 model ID 为 49，主流在不同 DP 中为 87 或 89；这些值是发现结果。
- 该形状条件为 24 本地专家、`5120→4608→5120` 权重接口及 top-6 输入。它支持主模型相容的重复执行段，不直接识别请求、源码层对象或运行时 replay ID。
- 其他同 model 流包含通信记录。N/A stream 的 task ID 并不唯一，不能当作另一条独立图执行相加。完整图与跨流关键路径仍需事件依赖。
- rank 0 / 8 的后续 `trace_view.json` 分别读取约 94 MB，通过该文件的同一导出时间轴裁剪到上述间隔。未将不同 CPU 采样轮拼接进这张时间线。

| 同条件统计（每 rank 中位后再取 rank 中位） | 初始窗 | 后续窗 |
|---|---:|---:|
| 主模型相容序列数 / rank | 5 | 12 |
| 主流段跨度 ms | 37.761 | 37.441 |
| 后续相邻段的间隔 ms | 不汇总混合期长间隔 | 35.740 |

后续窗口每 rank 11 个间隔，共 176 个。间隔内也有草稿形状相容计算、通信和 metadata kernel，不能整体标为空闲。rank 0 的间隔跨度中位 36.081 ms，全部 kernel 记录区间并集中位 7.349 ms；rank 8 对应为 35.341 / 11.187 ms。区间并集包括控制与通信标签，不是物理资源忙碌率。

## 主机范围内部的账

| 同次 worker trace | rank 0 | rank 8 |
|---|---:|---:|
| `prepare input` 范围数 | 12 | 12 |
| 完整范围时长中位 ms | 45.927 | 44.878 |
| 与相邻主模型段间隔相交的时长中位 ms | 34.006 | 31.485 |
| 每范围后代 cpu_op scope 数中位 | 6956 | 6956 |
| 未被直接子 scope 覆盖的根范围时长中位 ms | 16.590 | 15.403 |
| 同线程嵌套包含关系异常 | 0 | 0 |

完整范围长于间隔中的覆盖，说明原执行已经有主机准备与设备执行的重叠。exclusive 只表示没有嵌套记录覆盖，可能包含 Python/native 执行、等待、线程调度和 profiler 开销。导出 host JSON 的数值精度也不支持纳秒级物理解释。

一个代表性完整范围中有 976 次 `aten::slice`、1063 次 `aten::as_strided`、383 次 `aten::copy_`、245 次 `aten::where`、540 次 `empty_tensor`。这些是嵌套 scope 计数，不是独立 kernel 数；inclusive 时间不能叠加。4 个 `Event::synchronize` 的 exclusive scope 时间在两个 rank 的代表范围分别约 8.30 / 10.24 ms，也不等于可删除等待。

## 新候选 F33：A3 的批量 metadata 准备与固定输出接口

冻结源码 `model_runner_v1.py:2287` 的 `prepare input` 范围包含持久 batch 状态更新及 `:2495` 的 `_build_attention_metadata`。后者在 `:3827` 起逐 KV cache group 构造 metadata。源码已经有以下复用，不应重复登记成新优化：

1. 批次共享 `common_v41_batch_metadata`（请求/序列长度、RoPE、部分 SMLA/QLI 元数据）。
2. 同一物理 KV group 内的 `common_v41_metadata`，用于兼容布局的 LongKV/Indexer 坐标共享。
3. 相同 attention group 的层共享同一个 metadata 对象。

新的研究边界在 `dsa_v41.py:986–1035` 的非 A5 packed-cache 分支：按每个组的 `active_slots` 计算压缩完成判据、有效 mask、page 和 page 内偏移，并分别更新二维与 flat slot mapping。它由多次小张量运算组成。A5 有专用 backend 调用，不能假定 A3 已有同等接口；当前源还明确限制某些 graph 内 device metadata 功能需要 A5 packed-cache。

**具体重组：**以一个批次的多个独立 KV group 为外层维度，把 slot 的压缩/有效性/page/offset/flat-output 计算合并为批量设备算子或捕获的固定地址子图。共享输入只保留真实相同的请求计数与位置；每组保留自己的 block/slot、ratio、storage block size 和输出地址。这样研究的是减少重复 Python/dispatcher、临时对象与小任务边界，而非假设 KV group 的物理映射相同。

对于 group g 的原始 slot s，压缩分支先要求 `s≥0` 且 `(s+1)%ratio_g==0`，有效压缩 slot 为 `floor(s/ratio_g)`；C2 还受 actual-token/request 边界、position 奇偶及 skip-update 约束。有效结果写为 `(floor(slot/b_g), slot%b_g)` 和 flat slot，无效结果保持 `-1`。跨组批量执行只能合并这些独立计算，不能移动 compressor 状态更新、改变 sentinel、覆盖仍在图重放中消费的缓冲或混淆 SWA 与 LongKV 所有权。

**判定收益的必要指标：**同负载下未被前一段执行隐藏的准备时间、主线程 CPU ticks、每步 dispatcher/launch 数、metadata device 时间、最终客户端 TPOT/吞吐；还要检查抢占、错误输出、padding、空请求与 DSpark 接受/拒绝后的 cache 状态。仅把 CPU 操作换成一个 kernel、降低 CPU 利用率或减少 scope 数都不算端到端成功。

## 一次无 Torch profiler 的交叉检查及真实失败

run `worker-framework-20261030T105344Z-f556c759` 在本人常驻服务完成冻结场景预热与测量。使用每轮独立 cache_salt，让 seed 与对应测量共享缓存命名空间，避免旧请求使命中率变成 100%；没有 reset/restart 服务。

测量完成 8×32768 输入、8×1024 输出；命中/查询 235904/262144，0 抢占。客户端窗口 15.404247 s，聚合 531.801 输出 token/s。该轮尝试了外部采样，不能作为无采样基线，更不是相对于旧轮的优化收益。

procfs 20 秒采样完成，取请求业务窗口内两端样本之间 15.256 s：EP0 / EP8 主线程分别约 **0.968 / 0.966 核**；进程聚合约 1.917 / 1.880 核。主线程在该轮确实接近单核 CPU 饱和，但 procfs 不能归因到 metadata，也不能区分所有忙等。

py-spy 50 Hz 子进程持续报告采样落后，等待超时，最终均由本控制器 SIGTERM 清理（退出 -15），没有生成 speedscope 文件。因此本轮**没有有效 worker Python 栈**，不以预期的 prepare-input 栈作为结果。服务最终 health=200、两个 DP running/waiting 均为 0，进程仍属于本人服务。

低频复核 `worker-lowrate-20261030T105750Z-16d091d1` 将频率改为 1 Hz/12 秒，但在预热期间出现非本容器的 NPU PID 3418057，归属检查按设计终止本控制器，未进入测量。实际退出码 1；最终服务 idle，后续 health=200；没有停止或修改该外部进程。此尝试既没有有效栈，也没有用于比较的性能测量；后续只读检查时该 PID 已不存在，不把它记为持续阻塞。两次失败均保留 stderr 和退出码。

## 与文献重新对照后的优先顺序

1. **框架与 metadata 边界同时进入第一梯队。** FlashInfer 的 plan/run 分离与固定 workspace 地址提供设计方法；本机已有共享需要保留，重点改 A3 的具体准备工作。应优先验证未隐藏的边界，而不是笼统地“打开 graph/异步”。
2. **初始混合窗仍优先检查大行数 MoE。** Comet/FLUX 提供按消费者独立维度分块和重排的方法；本机 MC2 已融合并已有部分流水，不能再次计入整段 dispatch/compute/combine 融合收益。真实专家行数与内部阶段仍未知。
3. **后续算子机会继续保留。** QLI token 累加 K 特化、单组输出投影的二维接口、Q_b head 分片前移分别减少不同服务量。其收益与 F33 可能相互作用：主机准备缩短后，设备侧热点的端到端权重会提高。
4. **mHC 与 Engram 按测量落点研究。** mHC 已有 HcPre/HcPost；Engram 已见的大投影和 UVA 片段不能互相替代成本。文献的训练/GPU融合结果不替代 A3 真实执行验证。

当前证据没有给出任何已验证的端到端优化倍率，也没有给出不可超越的全机 TPS 数值。新增事实已经改变框架侧调查方向；下一步应针对 F33 的真实调用栈和未隐藏工作，而不是继续仅围绕 MC2 的 MAC 活动代理做推断。

## 复查入口

- [重复执行段原始分析](graph-executions.json)、[间隔裁剪结果](iteration-gaps.json)、[worker 范围及冻结源码](prepare-input-analysis.json)、[紧凑汇总](continuation-evidence.json)。
- [失败外部采样轮的有效计数、procfs、匿名请求时刻](worker-validation.json)，不包含 prompt 和生成文本。
- 3 个 `run_*analysis.py` / `run_graph_reanalysis.py` 从服务器现有导出重算；均由本机 ZCode DeepSeek 执行，主控核对退出码、源哈希与结果。高频采样失败另有 stderr、sampler exit 与最终健康证据。
- [完整阶段报告与历史边界](../publication-draft/README.md)、[本轮新增一手资料](LITERATURE-SUPPLEMENT.md)。
