# INT8 计算条件上界：来源、推导及证明缺口

> **读取原始比较文件前注意：**evidence/conditional-performance-comparisons.json 是保留的历史条件算术，其所列 F/P 和 throughput_upper_U 还需要对该短计时窗证明零初始突发或在途信用；仅有 sustained capacity 条件不够。一般式为 (F−β)_+/P，β 未证时这些数不是已认证硬上界。文件字段 measured_throughput 实为物理行数除以任务时长中位数，既未证明每行有效，也不是 token 吞吐或逐样本吞吐的中位数；不要与 PMU 活动比率等同。原 JSON 保持不变以保留证据哈希。

## 结果与范围

在下文列出的限制全部成立时，厂商估算代码对应的单逻辑设备 INT8 Cube 容量可写成 `P_est=353.8944 × 10^12 MAC/s`。对完整、标准稠密 `M=3、K=1280、N=32768` 的独立矩阵乘，得到条件计算时间下界 `0.355555... 微秒`。

**这不是已证明的 169 物理上界，也不是当前实际热态 kernel 的必要耗时，更不是整个推理系统的 token/s 上界。** 范围和未证前提见下文。

## 公式从哪里来

本地 `sources/primary/msprof-analyze/chip_peak_flops.py:40-44` 是华为工具 [固定提交的快照](https://github.com/Ascend/msprof-analyze/blob/6b8f431d4724043f44a39eef000fd5c6600784d8/msprof_analyze/cluster_analyse/cluster_kernels_analysis/operator_mfu/chip_peak_flops.py)。源码将 INT8 的 `OPS_PER_CYCLE` 写成 `16×32×16×2=16384 ops/周期/核`，以一个乘法加一次加法计为 2 ops；`:76-78` 将每周期 ops 乘以核数和频率。`:83-98` 从一个 profiler 的 `DeviceInfo[0]` 读 `ai_core_num`、`aic_frequency`。

冻结 `evidence/profile-device-capacity-info.json` 的 16 个逻辑设备条目各记录 `ai_core_num=24`、`aic_frequency=1800` MHz。将这些元数据代入工具公式：

`P_ops,est = 24 × 1800 × 10^6 × 16384 = 707,788,800,000,000 ops/s`；

`P_MAC,est = P_ops,est / 2 = 353,894,400,000,000 MAC/s`。

这只复算了源码的**估算值**。该函数在载入时选一个设备信息文件，并未在代码中验证 SoC 后缀、板卡资源共享、该类型指令物理最大发射速率、动态频率峰值，也没有证明该值对任何等价实现所可能使用的所有计算引擎都适用。官方工具的估计量、profile 信息的 `1800` 字段和当前设备物理硬最大值之间，不能直接画等号。

## 什么时候可以把估算量变成条件界

必须同时假设或单独证明：

1. INT8 每 AIC 的 `16384 ops/周期` 是这个具体 SKU、积累类型与稠密运算约定下的**硬上限**；工具公式或二进制中出现 `MMAD.s8` 不单独证明这点。
2. 所计 Cube 的频率从不高于 1800 MHz，24 核没有遗漏相关计算资源；profiler 元数据字段不单独证明这点。
3. 允许实现域中，所计的 `F_min` MAC 确属必要，而且全部由这一容量池执行；若计算可以放到 Vector、CPU、其他设备，需把这些路径的合法硬能力一起纳入。
4. 求算子界时，输出行均为有效、必要输出，输入、量化、精度、padding、缓存、共享和重排范围都已与实测严格对齐。求整轮输出 token 界还需证明所计 MAC 是该轮不可绕过的工作量。

在这些条件下，由资源容量得 `T ≥ F_min / P_MAC,est`。若无这些前提，`P_MAC,est` 只是名义计算器结果；不能称为实际不可超越的物理极限。

## 独立算子的精确算术

对严格限定的**标准稠密、全输出、逐个乘加都执行**算法，必要计算是 `F=3×1280×32768=125,829,120 MAC`。于是

`T ≥ F/P_MAC,est = 1/2,812,500 秒 = 0.355555... 微秒`；

`输出行速率 ≤ 3/T = 8,437,500 行/秒`。

这是 3 个矩阵输出行的条件速度，不能转换成模型有效 token/s。现有样本家族的 M=3、295 条记录和观测中位数 41.919 微秒见 `evidence/conditional-performance-comparisons.json`；但尚未证明这些记录满足完整有效行、相同算法与硬容量等全部限制，不应把二者的比值当成真实可达加速或正式性能差距。

## 多设备怎样计算，何时需要额外证明

如果对每个逻辑设备 `i` 都已证明同口径上限 `P_i`，而这组执行的实际工作量可无遗漏地划分到设备上，则**不要求设备彼此独立**也有 `P_total ≤ Σ P_i`。即便存在共享资源，求和仍是保守但可能很松的上限；共享资源的更强约束可再收紧。前提是每个 `P_i` 真的覆盖分配给它的全部操作，且没有漏掉其他执行路径。

例如 16 个设备在上述全部条件下的算术和为 `5,662,310,400,000,000 MAC/s`；这仅表示保守条件资源上限，不证明各设备能同时达到这个值，也不证明模型工作量可按之前的简化计数相加。DP/TP/EP、副本、空行、接受的投机 token、缓存复用和替代算法都必须在同一个工作量证明中处理。对计算、访存与通信等可同时进行的不同资源，应将有效时间下界取 `max`；只在证明数据依赖不可重叠时才可以将时间相加。

## 尚需资料与复算

先补官方产品/SKU 规格及实际 `socVersion` 到 ini 的映射、频率字段定义和硬频率上限、INT8 指令的每周期发射上限和真实计算资源边界；然后证明目标样本的有效数学语义和必要工作量。无法补证时保留 `P_max` 和 `F_min` 符号及 UNKNOWN。

运行 `python3 agents/bounds/int8_ceiling_certificate.py` 可以严格复算分数，并输出 `int8_ceiling_certificate.json` 与输入文件 SHA；脚本不验证上述证明前提、不访问设备、也不改目标服务。

## 已冻结 Q 投影采样的绑定强度审计（2026-10-29）

对 `tools/inspect_q_projection_allranks.py` 做静态复核：它从 rank CSV 筛选 `Type=QuantBatchMatmulV3`、前两项输入形状 `[M,1280]` 与 `[1024,80,16,32]`，并检查四项输入 dtype/format；再按同 rank、model、stream 的源窗口完整包含事件，汇总 `CAPTURED_TARGET_INTERIOR, M=3` 的 1920 条、时长中位数 42.0 μs。此**选择本身没有核对输出 shape、每条权重 tensor identity、NZ 布局与转置还原、有效行、每次量化对象、tile 或实际硬件指令**；同形状也不能将每条事件确定为 Wq_b 的同一调用。`1024×80×16×32=41,943,040=1280×32768` 是元素数算术相等，不是物理布局/输出语义证明。另有 `agents/fusion/f07-q-kv-streams.json` 的局部证据，但时间邻近和相容形状仍不能代替 tensor 身份。

该脚本直接读取目标窗口和 rank CSV，源码未见逐文件 SHA 验证，也未保存逐条事件起止键或输出形状；`evidence/q-projection-16rank-diagnosis.json` 指向冻结 manifest 本身并不能证明这一批读取时每个输入都与 manifest 相符。因而该汇总是有条件的采样线索，不能升格为逐事件已验证的工作量。它的 `0.355555 μs` 使用 `M=3,K=1280,N=32768` 完整直接稠密算法和名义 `353.8944 TMAC/s` 的共同假设；两者未与 1920 条实际 kernel 全部绑定，不能声称 118.125 是真实上界差距、利用率或可实现提速，也不能换算整模型 TPS。BF16 条件模型和 INT8 名义模型必须分开。

脚本 SHA-256 `75f4457f20a64e4ba26bb71c803b135ad66e2997b1aa89dba8f9da7e0339c6af`；归约结果 `q-projection-allranks/summary.json`、`evidence/q-projection-16rank-diagnosis.json`；复算仅用了已保存文件，没有 NPU 操作。若要缩小缺口，先查同一历史 task 的确切起止、output shape/dtype、权重 tensor/storage ID、NZ 逻辑布局与实际执行时的 hash/tiling 对应；找到并绑定 169 SKU 对应该 dtype 的全部资源硬速率、逐条必要工作之后，才可比较正式的受限界。缺一项继续 UNKNOWN。
