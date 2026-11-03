# MC2 低 MAC 比率：EP8 两个历史任务的分子分母拆解

## 新的有界结果

同一份带 profiler 的 prefill 采集中，选定 EP8 的第 14 层与第 39 层 `DispatchFFNCombine`；任务分别通过已存的 scope→flow→task→CSV 精确链，再按 task ID/stream/model/开始时间定位数据库 TASK；按各 TASK 的 globalTaskId 查询得到各 28 项 PMU，时间戳分别比所选 TASK 结束早 20 ns。开始时间先前的粗候选各两条，完整开始时间用于消歧。进一步检查 EP8 该次已保存数据库的整张 TASK 表（120292 行）：L14 的 globalTaskId 2538 和 L39 的 3506 各只出现一次；`TASK_PMU_INFO` 中每个 ID 各有且仅有 1 个时间戳组、28 条记录。结合两组时间戳均比对应 TASK 结束早 20 ns，支持这两组 PMU 分别与所选 TASK 关联；不过采集/导出器生成契约、计数器覆盖、时间戳含义仍未独立验证，不能把时间接近单独当证明或推广到其他 ID、其他数据库。具体关联和同次样本的局限见 [MC2-AIV-READ-VERSUS-POLL-20261029.md](MC2-AIV-READ-VERSUS-POLL-20261029.md)。

| 字段 | L14 | L39 | L14 / L39 |
| --- | ---: | ---: | ---: |
| TASK duration（μs） | 160365.956 | 8290.414 | 19.344 |
| PMU `aic_total_cycles` | 6788541718 | 335281950 | 20.247 |
| PMU `aic_mac_ratio` | 0.00259124 | 0.16437222 | 0.015764 |
| 数字乘积 `aic_total_cycles × aic_mac_ratio` | 17590740 | 55111039 | 0.319187 |
| PMU `aiv_mte2_ratio` | 0.125339 | 0.061819 | 2.028 |

**假定该两组 PMU 分别属于上述 TASK，且仅从所保存字段的数字运算看**，可以排除“这两条记录的 `cycles × mac_ratio` 数字乘积完全相等，低比率只来自 `cycles` 值更大”这一特定解释：两个乘积相差约 3.13 倍。反之，这不能排除真实计算工作相同而 PMU 定义、核归约、执行效率、采集误差等导致代理不同。数据来自单次带 profiler 采集，L14 和 L39 也未证明相同有效 token、expert 分布、实际工作或同一可执行对象，**乘积不是 MAC 运算次数、有效计算量、忙碌墙钟或可节省时间**。高精度转存不等于硬件测量精度，字段比率之间或跨核之间不能简单相加。AIV MTE2 非零说明此导出字段非零；不能单据它判断整卡不空闲、通信等待、HBM 饱和或特定操作的时长。

此前八卡已比较该两层任务与 AIV MTE2/MTE3 字段；本节补充的**完整 PMU 精度 AIC 乘积仅为 EP8 两条指定任务**，不推广到其他 rank、其他层或全负载。profile 时长不是未插桩基线性能；`cube_utilization` 也不能用作 MAC 比率的独立确认，既有分析指出所安装导出公式的口径和版本绑定有缺口。

## 在源码中能与不能解释什么

参考快照 `dispatch_ffn_combine_kernel.hpp` SHA-256 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`：

- 218–230 行：AIC 执行 GMM1、等待 AIV flag、GMM2；AIV 执行 `DispatchAndCombine`。因此这是一个融合任务，不应将整段任务时间全部当作矩阵计算时长。
- 450–529、561–620 行：矩阵乘的组内工作由每专家 `currentM`、`maxOutputSize`、scheduler 和每核循环决定，并含 cross-core 等待。只有在已绑定同一实现和运行参数时，这些分配才可能解释所测行为；物理输入形状不能代替有效专家分布。
- 693–703、801–866 行：参考 AIV 路径包含对 peer 计数状态槽的条件轮询、读取与每专家的远端数据搬运以及组同步；920–954 行还包含 GMM1 到激活、再到 GMM2 的跨核通知。轮询和正常 payload 读取都可能涉及 MTE2，因此当前聚合计数不能把两者分开；按组流水是否产生实际重叠和等待时长均未观测。
- 现有 [F21 分析](../fusion/F21-MC2-DISPATCH-ACTIVATION-PIPELINE.md) 已讨论源码允许的组级 overlap、等待计数、workspace 别名和细粒度重排的约束，不把已有 overlap 重报成新融合。

以上只证明**快照中存在这些代码路径**。历史 TASK 到此源码、编译对象、tiling 以及实际参数的绑定尚未完成；不能因此宣布这些路径确曾运行、跨卡通信是根因或该重排能提速。

## 下一步可判别证据

需要把具体调用与二进制/tiling 绑定，并取得同调用、同 rank 的有效 mask、专家行数及各组分配；若能找到已存在的同次内部阶段标记或可证明映射的 PC/指令轨迹，再区分状态轮询、payload 读取、空/少 tile 和跨核等待。现有采集的 `pcSampling=off`，已查数据库没有这种逐阶段轨道，故暂时不能量化各原因。只扩大同类聚合计数不能补齐任务身份与阶段对应；本阶段不新增 NPU 工作。

## 可复现

在任务根运行 `python3 agents/diagnosis/mc2_ep8_ratio_decomposition.py`。脚本验两份输入哈希、两层唯一性、task/stream/model/开始时间、PMU 数量和时间戳，再计算以上字段的比值；脚本 SHA-256 `baa0b7b8ed3414404f20a6f177f77bb7d8b5071c5a3c4df5c83db9c9b03038d7`。只读既有 JSON，不触卡、不改目标实现。

在同目录另执行 `python3 agents/diagnosis/audit_mc2_ep8_globaltaskid_uniqueness.py`，通过自有 stage 容器以 SQLite `mode=ro` 查询 EP8 的整张 TASK 表以及相同 ID 的 PMU 时间戳组，结果为 [mc2-ep8-globaltaskid-uniqueness.json](mc2-ep8-globaltaskid-uniqueness.json)，SHA-256 `90ff62966857c0d7d20c3fb10bc1721c12844a43f9431ab9d5e1ed35313144c7`。只查已有采集，不执行 NPU 负载。
