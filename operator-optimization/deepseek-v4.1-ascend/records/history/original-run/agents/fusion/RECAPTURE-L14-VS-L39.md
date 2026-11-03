# EP8：第 14 层与第 39 层提交任务对照

这是一次开启 Python stack 等 profiler 的采样，仅比较 EP8 `737-5` 的两个 target 层，不代表线上无 profiler 性能。两层同属于 runner 的 `8180` 调度 token、3 请求的 step；`1023` 是导出物理维度，不能直接等同实际有效 token 数。

关联链为：层 CPU scope 的 `call_id` 和 sidecar `layer_idx` → scope 内、同 CPU 线程的唯一 `torch_to_npu` start/finish → finish 的 `(pid,tid,ts)` 唯一设备 X → 按 `(start,model,task,physical stream)` 匹配 CSV。设备任务可在 CPU scope 结束后执行，这里的层号表示**提交归属**。第 14 层 scope 为 `737-20`，第 39 层为 `737-45`。定界的 flow 共关联设备 X 102 / 68 条，没有 X 匹配缺失或两层 scope 歧义；其中 CSV 精确匹配 62 / 36 条，剩余事件含 event/hcom，不能按未匹配视为不存在。其他提交线程、间接图任务也可能漏掉。

两层的 `DispatchFFNCombine` 导出输入都包含 BF16 `[1023,5120]`、两组 INT8 权重 `[24,5120,4608]` 和 `[24,2304,5120]`、路由相关 `[1023,6]`，输出 BF16 `[1023,5120]` 和 INT32 `[24]`。第 14 / 39 层各一条，stream 均为 47，flow ID 分别为 `1793241329140513188` / `1793241329850316287`；profile duration 分别为 **160,365.956 / 8,290.414 µs**。静态 shape 相同并不意味着有效专家 token 分配、远端到达、工作量、缓存和状态相同；这两条差距不能直接归因。两条中均只观察到单一复合 kernel 的总 duration 和导出 counter，没有内部 dispatch、专家 GMM、combine、同步等待或通信阶段的计时。不能把外部 EVENT WAIT 或 HCCL 数值与 MC2 做减法，作为其内部等待时长。

## 设备事件分布与等待范围

| 限定的关联事件 | 第 14 层 | 第 39 层 |
|---|---:|---:|
| EVENT WAIT 个数 / 单次中位数 / 最大值 | 16 / 201.276 / 32,276.415 µs | 12 / 331.273 / 12,516.870 µs |
| allgatherAicpuKernel，stream 92 | 1 / 232.115 µs | 1 / 3,226.136 µs |
| alltoallAicpuKernel，stream 92 | 1 / 376.633 µs | 1 / 409.412 µs |
| reduce_scatterAicpuKernel，stream 92 | 1 / 233.276 µs | 1 / 9,746.206 µs |
| MatMulV2 数量 / 单次中位数 / 范围 | 7 / 81.698 / 12.820–609.368 µs | 3 / 68.619 / 55.579–243.395 µs |
| InplacePartialRotaryMul 数量 / 单次中位数 | 6 / 29.969 µs | 3 / 33.379 µs |

EVENT WAIT 覆盖 stream 35、39、47；它是独立导出事件，不证明设备空闲，也未绑定到 MC2 内部的任何等待。上述 HCCL 类事件按 flow 归属于相应层提交范围，其通信组、对端、数据量、实际关键路径以及与 MC2 的内部关系均未闭合。通信事件、kernel、event 与其他流可能重叠，表里是单事件时长分布，不做加和排名。两层都观察到带约 `[1023,64,512]` Q 的 `SparseFlashMla`；第 14 层在本选定范围另有 QLI 同名事件、第 39 层未见，但不能把此范围未见解释为整层完全没有该工作。

## 已有候选的新的 prefill 条件化线索

在两层同一个设备 stream 47、各自 MC2 后的限定邻接事件，序列都出现：DynamicQuant → INT8 `QuantBatchMatmulV3` → `DequantSwigluQuant` → INT8 `QuantBatchMatmulV3` → `ForeachAddListV2`。第一个矩阵任务的 INT32 输出形状 `[1023,4608]` 与后续 `DequantSwigluQuant` 的 INT32 输入形状相容，其输出 INT8 `[1023,2304]` 与第二个矩阵任务的输入相容。**同流邻接和 shape 相容不证明是同一 tensor 或必然依赖**。

这是现有 **F22** 的一个具体 prefill 重新组合候选：若后续证明确为同一 gate/up 输出且无其他消费者，可研究矩阵乘完成一个所需 tile 后接入已有的 dequant/SwiGLU/quant，而非默认新建三份独立计算。候选边界需要保留 scale、clamp、激活及动态量化的完整归约/舍入和对外可见输出；逐 tile 提前消费是否可行取决于量化 scale 需要的整行归约、累加顺序、内存容量和 kernel 接口。`DeepseekV41MLP.forward` 在冻结源码 `model.py:144–148` 中明确是 gate/up → activation → down 的数据流，但尚未将这两层的设备 tensor/调用实例绑定到该模块。仅作为条件研究，不宣布已有融合缺失或可实现收益；`DequantSwigluQuant` 自身已经是导出融合任务。

具体可复核的邻接键（均 stream 47）：

| 层 | gate flow | MC2 flow | 首个 QuantBatchMatmulV3 flow | DequantSwigluQuant flow |
|---|---|---|---|---|
| 14 | `1793241329140467989` | `1793241329140513188` | `1793241329300908024` | `1793241329301010982` |
| 39 | `1793241329850277647` | `1793241329850316287` | `1793241329858636760` | `1793241329858737318` |

除了 F22，`HcPre` / `HcPost` 的导出维度与先前 F01–F03 的接口相容；`TransposeBatchMatMul` 的 `[8184,1,4096] × [1,4096,1024]` 与 F16 的输出投影候选接口形状相容。它们并未因此证明跨算子张量绑定，也不重复计为新融合发现。F21 的 MC2 已是复合任务；需更细的内核或通信身份/阶段计时才能判断该复合任务内部是否可重组、具体何处等待。

完整各名称的计数、最小/中位/最大单次时长、stream、同流邻接、所选 CSV shape/dtype 和输入哈希保存在 `diagnostic-20261029/ep8-l14-vs-l39.json`（SHA-256 `bf18660339d97eda36599f0f6e4d956b811b6cecc2f0deb2697c774e4c0fd8d8`）；该文件含用于本次只读查询的精确源码及其 SHA。只读取指定 rank 的已导出数据，未启动 NPU 或修改框架。
