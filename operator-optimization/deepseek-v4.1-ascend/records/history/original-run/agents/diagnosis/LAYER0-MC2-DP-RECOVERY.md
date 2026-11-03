# 第 0 层 MC2 的 DP 分组差异：恢复记录

## ratio 舍入的条件敏感性复核

对已保存 48 条第 0 层 MC2 记录，如果假设 ratio 按 0.001 四舍五入且 cycles 等其他输入没有误差，则先把每条 ratio 放宽 ±0.0005，再计算每组 8 rank 的 cycle 代理中位数，DP1/DP0 三轮 MAC 代理中位数比的外包区间分别为 `[1.1672,1.4029]`、`[0.9799,1.0218]`、`[1.0852,1.1633]`；MTE2 分别为 `[1.0959,1.1272]`、`[1.0162,1.0218]`、`[1.1127,1.1229]`。区间使用 DP1 下界/DP0 上界和 DP1 上界/DP0 下界，逐条限制在 [0,1]，属于保守外包，不表示端点都可同时达到。这些区间都小于对应任务时长中位数比 7.434、2.404、4.695。因此在这些假设下，ratio 的三位小数舍入单独不能消除两种倍率的差别。

采集 exporter 的舍入契约、周期含义及其他误差尚未证实；此计算不证明实际工作量相同，也不能区分任务内部等待、通信、缓存或路由。运行 `python3 agents/diagnosis/recheck_layer0_rounding_sensitivity.py` 可复现；脚本校验输入 SHA-256 并逐组检查 8 个不同 rank。没有运行 NPU。

已有三个内部采样轮次、16 rank 的已归属第 0 层样本中，DP1 的 MC2 任务通常比 DP0 长；各时长使用该卡自己的时间轴。按 ordinal 和 TP 编号排列不证明同一请求、collective 或对端关系。EP8 单样本已发现图任务 `globalTaskId/connectionId` 发生重放复用，PMU 与执行的时间近邻关联仍属候选；融合任务的内部 dispatch、GMM、combine、flag 等阶段没有分别计时，精确通信键无匹配也不能证明融合内部没有通信。详见 [README.md](README.md)、[loop-003-pmu-replay-conclusion.md](loop-003-pmu-replay-conclusion.md) 和 [16rank-mc2-other-stream-findings.md](16rank-mc2-other-stream-findings.md)。

## stage160：48 个固定任务的静态签名

先前仅记录输入物理首维 M=3；本次依照 16rank 已冻结窗口，按各卡自己的 CSV 以 `start/end/model/stream` 精确定位 48 条 `DispatchFFNCombine`。48 个 rank/轮次均唯一匹配，DP0 与 DP1 各 24 条，问题数 0。两组 48 条的完整**已输出字段**完全一致：

- 输入形状 `[3,5120]; [24,5120,4608]; [24,2304,5120]; [3,6]; [110592]; [122880]; [3,6]; [3]`，输入 dtype 分别为 BF16、INT8、INT8、INT32、INT64、INT64、FLOAT、BOOL，格式为 ND、FRACTAL_NZ、FRACTAL_NZ 和其余 ND。
- 输出形状 `[3,5120]; [24]`，dtype BF16、INT32；格式均 ND。导出核心类型为 MIX_AIC，block 字段为 24、mix block 字段为 48，状态为 static；CSV 的泛化算子名称 `DispatchFFNCombine` 哈希也相同。此名称不是二进制文件或其具体 kernel 符号的哈希，不证明编译对象及 tiling key 相同。

因此，在这些样本和这些导出字段的范围内，可以排除**两组记录的上述静态签名不同**作为直接解释；不能排除输入张量值、有效 token/mask、专家分配和路由、实际内存布局细节、权重驻留/缓存、内部运行分支或其他未导出属性的差异。同名同形也不证明通信参与关系、同一次 collective、同一个请求或相同数据工作量。合并内核的总时长和 counter 代理仍无法拆成网络、计算、等待，更不是可节省时间或端到端收益。

[layer0-mc2-physical-shapes.json](layer0-mc2-physical-shapes.json) SHA-256 `1bac00cd18dacce95b97fad47181ea776195b15a2ac964ecd1cc9a7d2aac8490`，包含逐条元数据、各组签名计数和限制；16 份 CSV 哈希通过它绑定的 stage102 记录核验。stage `operator-ceiling169-v2-160-mc2-signatures` exit-code 0、`model_verified=true`、模型 `deepseek-flash`；仅只读已有 CPU staging 数据，没有运行 NPU。

## 下一步的最小缺口

要判别动态工作不同、等待通信/同步、计算或缓存/带宽造成的差异，至少需要同一调用的有效 mask/实际 token、expert 选择或装载分布、可信内部阶段／等待及必要的关联键；现有静态 shape 和融合 TASK 的整任务指标无法给出这些值。只能在已有索引能将这些具体字段绑定到任务时继续有界查询，否则保留 UNKNOWN，不能靠扩大相同统计消除缺口。

## 第 0 层到 MC2 的已归属本卡局部区间

对冻结 `mc2-preceding-analysis/rank_layer0_timeline.csv` 中采样序号 1–3，按 DP、序号分组，每组 8 个不同 rank。先取各 rank **自身**已有区间时长，再取每组中位数，单位 µs：

| 采样序号 | DP | 第 0 层开始→MC2 开始 | 已归属 attention | FFN pre 开始→MC2 开始 | MC2 |
|---:|---|---:|---:|---:|---:|
| 1 | DP0 | 377.846 | 17.600 | 71.069 | 273.245 |
| 1 | DP1 | 383.772 | 17.220 | 69.789 | 2031.189 |
| 2 | DP0 | 394.722 | 37.950 | 70.978 | 392.173 |
| 2 | DP1 | 392.252 | 38.789 | 70.798 | 942.921 |
| 3 | DP0 | 386.582 | 39.290 | 70.698 | 353.973 |
| 3 | DP1 | 390.522 | 39.389 | 70.538 | 1661.877 |

前三列局部时长的两组中位数相近，而 MC2 中位数差异较大；这是已选数据的描述，表中中位数不必来自同一张卡。它不能判断不同卡的到达先后或第 0 层之外的准备时间，不证明跨 rank 同请求、相同输入/专家路由或共同 collective，更不能解释 MC2 内部阶段。时间未使用跨卡绝对时间戳相减；设备时钟频率误差及 profiler 扰动仍未校准。

复算 `python3 agents/diagnosis/summarize_layer0_local_lead.py`，脚本校验 6 组各 8 rank，源表 SHA-256 `d3e0d9cbeddcde97df34dcf36b386cc8201b71c82dde8ab4ed42dd713fde7dfb`。没有操作设备。

## Router 属性查询的判断

已有 [F20 源码分析](../fusion/F20-ROUTER-AND-DISPATCH-REVIEW.md) 指出相同 `MoeGatingTopKHash` 名称既可能带 `tid2eid` 也可能不带；当前缺相应导出算子的逐输入字段布局和动态张量值契约。仅扩查 CSV shape/未命名 `attr_N` 不能证明真实 token、expert ID 或动态路由，也不能仅靠名字确认 token 哈希分支。因此当前不为增加静态计数安排 router 查询；先要找到同版本算子输入/属性的可靠定义和与对应 TASK 的绑定，再评估是否有可区分假说的新证据。

## 动态 mask 与可用日志范围

冻结 `ascend_forward_context.py:203–232` 显示：在对应分支满足、预留 mask 可用时，代码从 `num_actual_tokens` 和 padded 数量构造 True/False 的 `mc2_mask`，padded 数量还可能依据跨 DP token 数确定；`moe_comm_method.py:594–608` 将 routing 的 mask 和 `topk_ids` 传入 MC2。因此 stage160 中 BOOL `[3]` 是物理输入形状，不证明三个元素均有效，也不证明两组专家分配相同。源码说明可存在的机制，不证明这些具体调用的实际值或运行分支。

本地保存的基线 `benchmark-20261028/raw-results/service-log-tail.txt`，以 `num_actual_tokens|mc2_mask|expert_token|ep_size|expert_parallel|collective` 固定关键词大小写不敏感检索，命中一条启动参数行，没有命中逐调用动态值；此 tail 不一定来自本次 profile，结论只适用于该文件及这些词。此前 EP8 profiler 字段/短标签检查的范围和无命中结果记录在 [request-dynamic-metadata-coverage.md](request-dynamic-metadata-coverage.md)，也不能推广到其他数据库或日志。没有读取或输出请求原文。

下一步须先取得**已存在且有来源索引**的记录，将图重放唯一身份、具体 rank 和调用与实际有效 mask/token 数、专家 ID/每专家 token 分布或内部阶段关联；当前既有摘要未提供这些字段，不按静态 tensor 形状冒充其值。若无这样的记录，本问题的动态工作量和等待原因保持 UNKNOWN，不做重复的大范围字段或日志扫描。上述 source SHA-256 `848112214c9f56371b4bf43a59d39fadbed0e0e864581e300a3a89778d4083cb`；基线日志 tail SHA-256 `64bfa3d8523ad86e1c466fcca166cdcfe1dc4a3676573fc51b65279a69aa8aba`。

## stage163：AIC 周期代理分组复核

对 48 条已归属 MC2，每条分别计算 CSV 的 `aic_total_cycles × aic_mac_ratio` 和 `aic_total_cycles × aic_mte2_ratio`，每组再取八卡中位数。表为每轮 DP1 中位数除以 DP0 中位数（**不是逐卡同请求配对比值**）：

| 采样轮次 | 任务时长中位数比 | MAC 周期代理中位数比 | MTE2 周期代理中位数比 |
|---:|---:|---:|---:|
| 1 | 7.434 | 1.284 | 1.112 |
| 2 | 2.404 | 1.001 | 1.019 |
| 3 | 4.695 | 1.124 | 1.118 |

独立复核把 timeline、stage102、stage159、stage160、calls 五份已有记录逐样本按 rank/内部轮次核验，48 条的 `start/end/model/stream/taskId` 一致，并重算了六组各八卡的中位数。stage163 原脚本本身未独立校验 start/end，上述补充核对仅证明这些冻结中间记录彼此一致，仍受源 CSV 采集和归属规则限制。

在这个已导出代理和分组范围内，两种周期代理的中位数倍率比任务时长倍率小得多；不能据此说两组真实 MAC 次数或搬运字节相同、差出的时间全是等待通信、或预测可消除收益。`ratio` 在 CSV 中只显示有限小数，DP1 的 MAC 比率低至 0.006，量化/舍入对相对差异影响可能很大，且本次没有已证的四舍五入契约。MTE2 活动代理不是 HBM 字节或带宽饱和指标；不同硬件流水与周期分母也不可直接相加。三个采样轮次、八卡是同一次 profile 采集，不是独立运行或已证的跨组同请求。

已保存的 stage159 AIV Vec 代理在 DP1 出现 0.000/0.001 等精度很低的报告值，直接比较不能判定 AIV 没有工作。若后续考虑原始精度的 PMU 来限定此不确定性，须先固定 graph replay 关联：EP8 既有证据表明 `globalTaskId` 可复用，PMU 时间接近 TASK 结束只能提供候选而非已证明唯一键；也须核实各项 counter 的版本语义和采集范围。

[layer0-engine-work-proxies.json](layer0-engine-work-proxies.json) SHA-256 `9738b889bdf3f8d418f5bb1e9c2bd83c4c07f7c32e36e62c9c1c08482258f860`；由已保存源表及 [复算脚本](recheck_layer0_engine_work_proxies.py) 生成。没有新 NPU 负载。

### 原始 PMU 对重放的归属止点

复查 [loop-003](loop-003-pmu-replay-conclusion.md)、[loop-004](loop-004-pmu-export-source-conclusion.md) 及已有 [Engram PMU 字段核查](ENGRAM-MATMUL-DISPATCH.md)：EP8 的同一 MC2 `globalTaskId=18529`、`connectionId=184330` 在 7 条不同时间的 TASK 中复用；7 组 PMU 时间戳接近各自 TASK 结束与一个 CSV 的数值一致，仍只是候选，缺少已验证的时间戳生成规则、允许误差或独立 replay 键。已检索的同镜像 849 个 Python/SQL 文件中没有发现该逐次 PMU 归属契约；范围不含二进制、其他拼写或未保存来源。Engram 的其他任务也观察到 ID 复用及相同缺口，不构成 MC2 的新键。

因此当前**未发现能将 MC2 原始 PMU 唯一绑定至具体重放的已验证新键**，不继续扩查或把时间近邻提升成事实。需同版本导出器的时间戳／replay 契约或新的明确来源证据才能重启这条路径；此前保留的 CSV 代理仍受其小数精度限制。

## stage165：EP 与 MC2 分组构造

只读获取基线容器与 CPU stage 容器的两个固定源码文件，各文件在两容器的 SHA-256 相同：`vllm/distributed/parallel_state.py` 为 `d9b10ad56833ebc08a753e1a050ef0989c30d4134021b7b7b00150b7bd01adc6`；`vllm_ascend/distributed/parallel_state.py` 为 `b7ea7d6fb74530c50aec718d34266bae952ed96cc5a85766d7e208b952b10445`。来源的初始化逻辑以 ExternalDP×DP×PP×PCP×TP 排列 world ranks，上游 EP 组和 Ascend 单独创建的 MC2 组均按 DP×PCP×TP 形成组成员列表。

**条件性成员结论：**若该 profile 的 actual world=16、DP=2、TP=8、PP=1、PCP=1，源码在运行时按这条初始化路径生效，且 rank 映射无其他变换，则 EP 成员列表和 MC2 成员列表均为 global rank 0–15 的一个 16 人集合，DP0 与 DP1 位于相同的组成员集合。EP 与 MC2 是分别创建的组对象，即使成员相同也不能称同一个 communicator。冻结启动命令的 DP2/TP8/EP 开关、profile 文件命名中的 EP0–15 和另一服务日志中的 EP0/16、EP8/16 支持这组条件，但另一次日志进程与 profile 的进程不同；actual PP/PCP/world、真实 group 对象和 rank 映射尚需本次运行记录绑定。相同组成员不证明两张卡选中的任务属于同一次 collective、相同请求或等待因果。

另一个尚未闭合的路径：`moe_comm_method.py:594–608` 的 `dispatch_ffn_combine` 接收 `token_dispatcher.moe_all_to_all_group_name`；不能仅据存在 `get_mc2_group()` 就断言该调用传入哪一个组。后续需读取该 dispatcher 的固定同镜像源码及构造条件。

[ep-group-source.json](ep-group-source.json) SHA-256 `ae2deb3834d57dea18d5e31feadb586d4bf3328c03790341aa97e17c1e6c83c2`；stage `operator-ceiling169-v2-165-ep-group-source` exit-code 0、`model_verified=true`、模型 `deepseek-flash`。只读两个容器源码，没有运行 NPU。

### stage166：具体 dispatcher 的组名来源

两个容器中的固定 `token_dispatcher.py` SHA-256 均为 `1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb`，`moe_comm_method.py` 均为 `7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7`；后者与本机冻结源码的哈希也一致。源码 `FusedMC2CommImpl._get_token_dispatcher` 创建 `TokenDispatcherWithMC2`；其构造使用 `get_mc2_group().device_group`、组内 rank 和该组 backend 的 `get_hccl_comm_name`，并从该 MC2 组取 world size。具体 `dispatch_ffn_combine` 调用传入上述 communicator name 和 world size。

因此，在目标 profile 的该算子调用确实执行此源码路径、使用此 dispatcher 实例的条件下，其 group 参数绑定到 **MC2 组的 communicator**，不是上游 EP 的 communicator。按 stage165 的 world/DP/TP/PP/PCP 初始化假设，这两个独立 communicator 的成员集合相同；仍需本次运行时对象、实际参数和 rank 映射证据才能把条件结论提升为运行事实。共享成员不能识别特定调用或说明一方等待另一方。

[mc2-dispatch-group-source.json](mc2-dispatch-group-source.json) SHA-256 `5cd7cb19eec0ad115c85bdd1b78c93f160f84776115d8d3c5a831b54f184983e`；stage `operator-ceiling169-v2-166-mc2-group-source` exit-code 0、`model_verified=true`、模型 `deepseek-flash`。双容器只读源码，无 NPU 负载。

### stage167：两个图任务的属性数据

精确匹配 EP0、EP8 的第 1 轮第 0 层 `DispatchFFNCombine` TASK 后，二者均可解析 `COMPUTE_TASK_INFO.attrInfo`，整段属性内容的哈希相同；解析出 `attr_0` 至 `attr_5` 六项，逐项值哈希也相同。**没有显式 `group`、`group_name`、`world_size` 或 `rank_size` 标签。**属性的原文没有输出。

冻结 tiling 源码中算子 ABI 的 index 0 是 group、index 5 是 world size，但没有找到 profiler `attr_N` 与该 ABI 索引一一对应的契约；不能将这些位置直接解释为组名或成员数。图中的 `globalTaskId`／属性可跨 replay 复用，相同属性哈希也不能证明两次执行同属一个 communicator 实例、同一 collective 或同一请求。这个小范围结果没有把 stage165–166 的运行条件升级为已观察的实际组成员；已有精确通信键缺失不代表内核没有通信。

[mc2-group-attr.json](mc2-group-attr.json) SHA-256 `a3f4e5f78ac1724e5a816a4da1b7d4fc2a1bb889078697a06b163200c690ee60`；stage `operator-ceiling169-v2-167-mc2-attr` exit-code 0、`model_verified=true`、模型 `deepseek-flash`。只读两份已冻结 profiler DB，不读取通信表、不使用 NPU。

### stage168：profile 元数据中的分组入口

复核 EP0、EP8 已保存 DB 的完整 26 表 schema，未见显式 communicator 成员表；`RANK_DEVICE_MAP(rankId,deviceId)` 只能说明设备映射，不能单独代表组成员。两库的 `META_DATA` 各 8 行，在固定白名单与限定 JSON 遍历范围内，各发现一行名称中含 parallel/group、结构内有三个 `group_name` 标签；该检查没有把 `global_ranks` 和 `group_rank` 纳入白名单，不能把未输出成员解释成不存在。rank/device 映射分别记录 0→0、8→8；非 JSON、其他标签、截断范围外及其他库或日志不在结论范围。

已冻结 `torch_profiler_interface.py:372–412` 中，`parallel_group_info` 将已初始化 HCCL 组按 communicator name 建字典，内部记录 `group_name`、`group_rank` 和 `global_ranks`。这是可继续核查的明确来源入口：对两库固定元数据行按其内容哈希提取这三个字段，按 comm name 哈希与 stage167 属性做**字符串相等候选**比较。还需确认 exporter 版本和数据覆盖；profiler 的 `attr_0` 到算子 ABI group 的映射未证，哈希相等不能自动证明是同一 communicator，更不能证明同次 collective。

[mc2-profile-topology-metadata.json](mc2-profile-topology-metadata.json) SHA-256 `195aeec3aa1377c95655ed07a7e9c867b283d1187bbd842664a9e0947d6bff16`；stage `operator-ceiling169-v2-168-profile-topology` exit-code 0、`model_verified=true`、模型 `deepseek-flash`；无 NPU 负载。

## stage163：周期活动代理的独立核验

`python3 agents/diagnosis/recheck_layer0_engine_work_proxies.py | cmp - agents/diagnosis/layer0-engine-work-proxies.json` 在任务根目录逐字节通过。脚本关联 48 条 timeline 与 rank/target forward ordinal 的 layer-0 记录，并对照 stage159 的 rank/run、task ID、stream、时长和 MAC ratio。独立附加核验：48 个 `(rank,run)` 唯一，逐条 timeline 的 MC2 开始/结束纳秒与对应 `rank_target_calls.csv` 开始/结束均完全相等。仅靠可重放复用的 task ID 不足以识别执行；此处的本卡时间窗口与 ordinal 检查提高关联可信度，但不证明跨卡共同 collective 或同一请求。stage159 从被 hash 绑定的各 rank kernel CSV 中选择 model 49、落在对应窗口的记录；本次未重新读取远端 CSV。

定义每条记录的 `MAC 代理 = aic_total_cycles × aic_mac_ratio`，`MTE2 代理 = aic_total_cycles × aic_mte2_ratio`。两者都是以导出 cycles 字段乘以无量纲导出比率得到的 **cycle 口径乘积**；其可解释为对应活动周期近似，需以当前采集版本的分子、分母、MIX 计数和采集覆盖确实对应为前提。本次尚未证明导出器具体实现完全匹配相关官方文档；乘积不是 MAC 指令/运算数量、搬运字节、传输带宽，也不是全部资源忙碌周期。AIC/MTE2 重叠、多个 core 聚合和时钟口径均不能仅由乘积还原。

| 采样序号 | DP1/DP0 时长中位数之比 | DP1/DP0 MAC 代理中位数之比 | DP1/DP0 MTE2 代理中位数之比 |
|---:|---:|---:|---:|
| 1 | 7.434 | 1.284 | 1.112 |
| 2 | 2.404 | 1.001 | 1.019 |
| 3 | 4.695 | 1.124 | 1.118 |

每组各 8 条；逐条先算乘积、在各组分别取中位数，表中是**两个组的中位数相除**，不是逐 TP 匹配后的比值中位数，也不是总体合并比例。就这三个已选样本轮次而言，组间代理中位数比值小于组间任务时长中位数比值；这不是同一工作量从 DP0 到 DP1 的变化率，不证明实际计算/搬运相同、只增加了指定百分比、等待是原因、或该时间可消除。三轮相关、可能受相同 replay 和采集扰动影响，不支持总体统计推断。

CSV ratio 显示为有限小数，采样、导出、舍入及时间/周期口径的误差上限本次未知，不能把乘积展示的小数位当测量精度。仅为解释量化风险：若已另证 ratio 精确按 0.001 四舍五入，显示 0.002 对应约 `0.0015–0.0025`，乘积的这项相对量化不确定度可达约 25%；实际舍入方式和其他误差尚未证实，因此此数不是已证总误差界。倍率比较是对导出值的描述，不是误差修正结果。

输入/复算脚本哈希见 `layer0-engine-work-proxies.json`；该结果 SHA-256 `9738b889bdf3f8d418f5bb1e9c2bd83c4c07f7c32e36e62c9c1c08482258f860`。本次复核仅使用已保存资料，未启动 NPU。

### Stage 169：两份 profile 已保存的通信组元数据

- 本机 ZCode 执行记录 `operator-ceiling169-v2-169-profile-groups.exit-code` 为 0，`verified.json` 为 `model_verified=true`、`deepseek-flash`。结果 `agents/diagnosis/fixed-profile-group-members.json` SHA-256 为 `84ed981f8b6b1f5ea45677eedfaa794052f2874a20f367da2d12da585b2e6e2c`；脚本 SHA-256 为 `ce17b32efe100c6c2a24761c282db5c1c813cd52dc15bf74452f92125522215c`。脚本将两个固定 DB 和 metadata 内容绑定 stage168/manifest 哈希，并以 SQLite `mode=ro`、`query_only=ON` 读取。
- 两份 profile 各有三个明确标注的条目：EP0 的 TP 成员是 0–7、组内 rank 0；EP8 的 TP 成员是 8–15、组内 rank 0。EP 与 MC2 条目各自都列出全局 rank 0–15，EP0 和 EP8 的组内 rank 分别为 0 和 8；按列表索引均对应各自 `RANK_DEVICE_MAP.rankId`。两个库中同名 EP 条目的通信键哈希相同，MC2 条目的通信键哈希也相同，且 EP 与 MC2 键哈希不同。这是两份采集文件所记录的成员，不能延伸为全部 rank 文件都已检查。
- 两库中标为 MC2 的元数据字典键 UTF-8 SHA-256 均为 `5bab99054be7eecdccd4a00b7191bf9b99af628679092ab30cb583da223f4ecb`，与 stage167 两条 TASK 的 `attr_0` 值的 UTF-8 SHA-256 相同；两个提取脚本均对 Python 字符串使用默认 UTF-8 编码，因而是同一编码下的强字节匹配线索。TP、EP 未匹配。stage167 的 profiler `attr_N` 到算子 ABI 参数及其序列化契约尚未证实，匹配本身不足以将具体 TASK 或 replay 确证绑定该 MC2 通信组。
- 冻结的 `torch_profiler_interface.py` 描述了可能的生成契约：以 `get_hccl_comm_name(..., init_comm=False)` 结果为键，值包括 `group_name`、`group_rank`、`global_ranks`；本次实际采集/导出是否使用完全相同版本仍未独立证明。stage166 的源码提供 MC2 dispatcher 调用 MC2 组的条件路径，但不能单靠源码和哈希完成实际 TASK 的绑定。
- 相同成员或通信键都不证明两张卡的样本对应同一请求或同一次 collective，更不能证明谁等待谁、等待持续多久；未做跨卡绝对时间比较。后续如要建立运行时绑定，需带版本契约的 profiler 参数映射或明确调用/运行时 ID，而非再次比较已有无标签属性。

### stage169：group 元数据和 attr 哈希的独立核验

两条 profile 样本与 stage167 通过完整 rank 名称和各自的 profiler DB SHA-256 对应。两份 `parallel_group_info` **直接记录**：各有一个 group_name 经 allowlist 正则归类为 `mc2` 的条目，它们的 `global_ranks` 均为 `[0,1,...,15]`，`group_rank` 分别是 0、8；对应的 communicator key 摘要和各自第 1 轮第 0 层 `DispatchFFNCombine` 的 `attr_0` 摘要均为 `5bab99054be7eecdccd4a00b7191bf9b99af628679092ab30cb583da223f4ecb`。两份记录的 TP 列表分别为 0–7、8–15，两个 EP 类记录的列表均为 0–15，但仅凭此不能确认其他 14 份 profile 元数据或各设备物理互联拓扑。

**实际计算摘要的方法必须区分**：stage167 `query_mc2_group_attr.py` 从 `STRING_IDS` 取属性字符串，以 `|` 分项、以首个 `:` 分割标签与值，再对提取的值执行 Python 默认 UTF-8 `.encode()` 后 SHA-256；没有 trim 或额外 Unicode 归一化。stage169 `query_fixed_profile_group_members.py` 先以 `json.loads` 解析 `META_DATA` 的 `parallel_group_info` 文本，再对解析后字典 key 执行默认 UTF-8 `.encode()` 后 SHA-256；JSON 文本中的转义已由 JSON 解析。两侧没有输出原文。相同 SHA 是**经各自解析后的 UTF-8 字节高度匹配**，在正常碰撞假设下支持两个解析结果相同；不证明两份数据库原始序列化字节相同，也不单凭哈希证明活体 communicator 是同一对象。`value_length` 是 Python 字符数，不保证等于 UTF-8 字节数。

冻结 `torch_profiler_interface.py:372–412` 描述一种可能的生产路径：遍历已初始化 HCCL 进程组，使用 communicator name 为 key，存 `group_name`、`group_rank`、`global_ranks`，查询失败时可能缺项。当前没有独立证明历史 profile 的确由该源码版本、完整注册和相同编码生成；它解释这些字段的可能含义，不能单据它证明生产链及所有成员已经执行某个请求。

相同 baseline/stage 哈希的 `token_dispatcher.py` 中，`TokenDispatcherWithMC2` 获取 `get_mc2_group().device_group`，调用 backend 的 `get_hccl_comm_name(local_rank)`，把该名称与 `get_mc2_group().world_size` 保存；`moe_comm_method.py` 在相关分支中将它们作为 `dispatch_ffn_combine` 的 `group` 和 `world_size`。producer 示例传给 `get_hccl_comm_name` 的是 `global_rank`，dispatcher 源码传的是组内 `local_rank`；现有资料未核实 API 参数语义或两路径运行时身份。两侧名称摘要相同虽是关联线索，但 profiler `attr_0` 与算子 ABI 的 `group` 字段映射/序列化未证，不能说该任务确定使用该元数据组，不能把两任务认作同一次 collective，也无法推断跨卡到达顺序、网络耗时、等待原因或 16 个成员当次有效专家路由。

本机独立计算核验：两边 source JSON 内 `token_dispatcher.py`、`moe_comm_method.py` 内容的 SHA 均等于记录值；stage169 与 stage167 的 DB 摘要匹配，两侧各只有 `mc2` 类 key 的摘要与对应 `attr_0` 相同。`mc2-group-attr.json` SHA-256 `a3f4e5f78ac1724e5a816a4da1b7d4fc2a1bb889078697a06b163200c690ee60`，stage167 查询脚本 `8cfeec1616dafbfd0c489070ee24418c3e7d656900f77dc962d2eee8d7c87e80`；`fixed-profile-group-members.json` `84ed981f8b6b1f5ea45677eedfaa794052f2874a20f367da2d12da585b2e6e2c`，stage169 查询脚本 `ce17b32efe100c6c2a24761c282db5c1c813cd52dc15bf74452f92125522215c`。本次只读已有结果和源码，不触卡。

### 首层起点与图执行的既有记录：限定重算及时间边界

复用 `model-execute-wait-chain-v2/calls.json` 和 `mc2-preceding-analysis/rank_layer0_timeline.csv`，仅按完整 rank 与该 rank 的 target forward ordinal 关联采样序号 1、6、7 的 48 条。每条的既存导出链按 connectionId 和 modelId 匹配一个 `MODEL_EXECUTE`、一个 `MODEL_WAIT_COMPLETE`，每条对应首层起点位于该 wait 记录区间内。归属除显式导出键外依赖时间包围；没有证明运行时 model handle 或请求身份。前一表与后一表来自不同导出视图，以下相减只在这两类时间戳已经映射到同一设备时间轴且误差可忽略的条件下解释，**不是已认证的因果耗时**：

| forward ordinal | 组 | 记录中 wait 起点→首层起点的区间，中位数（µs） | 最小–最大（µs） |
|---:|---|---:|---:|
| 1 | DP0 | 4985.848 | 89.478–7321.974 |
| 1 | DP1 | 5005.410 | 82.639–6226.416 |
| 6 | DP0 | 308.874 | 90.120–330.413 |
| 6 | DP1 | 301.947 | 84.658–326.533 |
| 7 | DP0 | 377.462 | 85.118–400.680 |
| 7 | DP1 | 305.307 | 85.978–322.313 |

各组 8 个本卡差值的中位数、范围均分别统计；没有跨卡时间戳相减。ordinal 1 两组中位数相近、各组离散较大，ordinal 6 中位数相近，ordinal 7 DP0 中位数较大；不能由这些分组数据排除输入、图前准备或调度差异，更不证明区别来自这些因素。`MODEL_WAIT_COMPLETE` 是导出标签，其包络不等于 CPU 等待、设备等待或实际前置任务，并非首层真正依赖起点。若想归因须获得同次重放关联、首层输入 producer、等待 event 对象/信号及版本内同卡 TASK/CSV 时间映射的可信契约。

复算 `python3 agents/diagnosis/recheck_layer0_replay_lead.py`；输入 SHA：`calls.json` `ecb1ba603e52cf3cf6722f4b8cb0c1aad3b295c596f4f60c7fc047ecff6ef925`、`rank_layer0_timeline.csv` `d3e0d9cbeddcde97df34dcf36b386cc8201b71c82dde8ab4ed42dd713fde7dfb`。固定脚本检查目标各组 8 条及唯一 wait、包含关系。只读历史数据，未运行设备。

此前 `evidence/clock-source-and-anchor-audit.json` 重建 16 卡的非同一时刻 anchors，并保存同版本候选转换源码；`evidence/g4-clock-bound-integration-audit.json` 明确物理跨卡时钟误差界未验证。device anchor 偏移的 9520 ns 极差是元数据观察值，不是物理同步误差上界，也不足以校准漂移及采集延迟。故既有跨卡绝对到达/下发时间差及跨 DP ordinal 对应均保留条件，不用于断言先后、等待或因果；可靠补证需要实际导出路径、统一时钟域及采样误差/漂移界，且另需跨 rank 同次调用身份。现有资料无已证这两项的入口，不安排重复查询。

### MC2 内部阶段与可用于判别动态路由的输出入口

已保存源码 `dispatch_ffn_combine_kernel.hpp`（SHA-256 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`）定义 AIC 走 `GMM1 → CrossCoreWaitFlag(SYNCFLAGV2C) → GMM2`，AIV 走 `DispatchAndCombine`。后者执行 active mask、routing、跨 rank token-per-expert 通知/检查与累计，随后输出经 rank 维累计的每本地专家计数，再向 AIC 发 cross-core flag；后面 AIV 的 SwiGLU 还等待 GMM1 的 cross-core flag，再通知 GMM2。GMM1/GMM2 中每个专家的 `currentM` 来自累计数组对应项，并受 `maxOutputSize` 和此前各专家总量截断；仅当该专家有对应计算块且 `currentM > 0` 才运行矩阵块。这为 AIC 低活动提供**可证的源码机制候选**：动态专家计数、跨核心通知、跨 rank 信号检查都可影响内部执行；不证明它们在本次任务消耗了多少时间，或是组间差异原因。

这个实现的 `expert_token_nums` 输出在内核源码中由 core 0 从 `cumsumMM[(EP - 1) * expertPerRank]` 开始复制 `expertPerRank` 个 INT32；`GetCumsumForMMAIV` 沿 EP 维累计计数；前述 GMM 也引用该累计数组。保存的 `moe_comm_method.py`（SHA-256 `7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7`）在该 `dispatch_ffn_combine` 分支把预分配 `self.expert_token_nums` 传入并作为 `expert_tokens` 返回。stage160 观察到该任务输出之一为 `[24] INT32`，形状相容，但未证明该输出槽与源码变量一一绑定，也未证明历史采集使用这份具体实现。相同文件的 CANN MegaMoE 是另一分支，不应将两者等同。

若历史记录中确实存在**该次、该 rank、该层、该次 graph replay 完成后且覆盖前**的 `expert_token_nums` 值，且可验证实现路径、输出槽及 `maxOutputSize`，可按源码专家顺序和截断规则重建 AIC 的每专家 `currentM`，进而检验两组专家负载是否不同。该 24 项计数不能独立量化跨 rank 等待，亦不能反推出 token 文本或完整 peer×expert 计数矩阵；也不能以当前共享缓冲区内容代表过去某次调用，因为源码显示它是预分配的实例缓冲。本次不访问设备缓冲。

`evidence/used-raw-source-sha256.json`（SHA-256 `cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a`）是 97 个**已使用**来源而非完整文件清单，路径名中无 dump、tensor、routing、expert；这不是历史不存在 tensor dump 或相关日志的证明。此前动态元数据检查见本记录“动态 mask 与可用日志范围”，没有逐调用的计数。当前未找到已保存且可绑定的值，不追加只读检索来重复统计静态 shape；动态负载、内部 AIC/AIV 等待均保持 UNKNOWN。下一入口应是有来源索引的既存张量 dump/按调用专家统计；如果没有该入口，需将可观测证据缺口留给后续采集方案，不推测数值。

对可能的 Engram 重叠，stage102 的 48 个 MC2 窗口只在其他数字 stream 匹配到 28 条 `allgatherAicpuKernel`，第一轮 16/16 无这类已导出重叠；这排除的是**已导出数字 stream 中的该类记录必须与每一条慢 MC2 同窗**，不排除 CPU 卸载、未导出设备工作、同流活动或 MC2 内部通信，也不证明任何资源争用或独立请求。来源和并集覆盖限制见 `16rank-mc2-other-stream-findings.md`。

### 已存专家统计入口与条件性的 tile 检查

对本机已保存基线 `benchmark-20261028/raw-results/results/{warmup,measure1,measure2}/metrics-{before,after}.txt` 共 6 文件，仅匹配 Prometheus `# HELP`/`# TYPE` 行的指标名中 `expert|moe|routing|eplb`（忽略大小写），均无命中。SHA-256 按上述文件顺序分别为 `6bc568b0de211ceac48d5a6d7d71f5b12be3b2b4b54db7bd7465340d9343d06f`、`70aee3a81f3192f0d363c473351d7bdce60e8dfd74abe42280f29a4dfb5059b8`、`d9dacde0fbdd5d82a4f37b2f4bfde5501f5a5109b54dc7c70b753f3dd50c7df8`、`e1bfaffa59f9cdfe989f908c90b0e6e7c678c4a63d69377d78014b8fb8d1ea6e`、`df4760abb2e74ecaf8f89e3abbe7f4345d77349d5b69b53eaa5ec0426d29e5b0`、`9db394fab194048e381262e1fefa741f92aac2d8c73fa083c3c95f88a10dd1ec`。这只覆盖标准 HELP/TYPE 名称且只检索上述词，不能排除其它命名、缺 HELP 的指标或其他日志；它们也不是目标 profile 的逐调用统计，即便有进程级汇总也不能凭近邻赋予某次 MC2。

`evidence/profiler-export-inventory.json`（SHA-256 `ff0f4c8a7e46a48ca53f376dc749406fdced163de9cdca000c7681ff41b592f7`）记录 16 个 rank 的导出目录文件，所列路径无 `expert|routing|tensor|dump` 字样；路径名检查不读取内容，不代表文件内部不含数据、未列目录无 dump 或相关统计从未产生。前述已使用文件索引、已检服务日志尾和相关元数据的限制仍然适用；尚未发现能绑定目标 rank、layer、replay 的专家计数值。

**独立可证伪条件模型：** `dispatch_ffn_combine.h` 中本实现 L1 tile 的 M 为 128，GMM1/2 在 `currentM <= L1TileShape::M` 时为各专家权重设置 `CACHE_MODE_DISABLE` hint。假定某个共同 MC2 调用实际有 16 个参与 rank、各 rank 至多 3 个有效 token 行，并且每 token 对同一个专家最多投递一次，那么对每个专家，总接收行数上限为 `16 × 3 = 48`；经非负容量截断后 `currentM ≤ 48 ≤ 128`，因此源码中该条件成立。没有去重条件时，6 个 top-k 槽所给的朴素上限是 `16 × 3 × 6 = 288`，不能从它推出小于 128。后者上限同样以实际参加者、每卡最多 3 行及每槽至多贡献一行为条件。

stage160 的所有 48 个采样记录各有物理 M=3、top-k 形状 `[3,6]`，stage169 的两份 group 元数据列出 0–15；这些事实**不证明对应 16 个所选样本是同次 collective，也不证明这些就是该次调用每个 peer 的输入**。`fused_topk_router.py` 的一种动态路径使用 `torch.topk(...).indices`，但 vision/hash 路径可使用 `tid2eid` 表，融合 `moe_gating_top_k_hash` 的去重契约及历史运行分支尚未证明；故实际的逐 token 去重、48 上界与 hint 是否应用均为 UNKNOWN。即使分支成立，软件 hint 也不等于已证的 L2 未命中、HBM 字节、带宽瓶颈或具体耗时原因。关联源码 SHA：`dispatch_ffn_combine.h` `d2b5712ad97f7505233f340e18660627424e5e1fe4c33be68f8e34b51a479ab1`；router `06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838`。未运行 NPU 或改服务。

### 已存二进制与任务绑定的限制

stage160 `query_layer0_mc2_physical_shapes.py` 计算的 `kernel_name_sha256` 实际是 CSV 的 `Name` 字符串哈希：对 `DispatchFFNCombine` 做 UTF-8 SHA-256 得 `6e8e5bd2d493e5c51d99197f8b01a629eb82c9673cea6601476b7e059eb7561f`，与 48 条结果相同。这不是编译对象文件 SHA，也不是特定 symbol/tiling 的身份。源脚本 SHA-256 `5921775c1cc9bfb69525ad681d7d6af17c885f6832d08a40f5d7855e3cafb308`。

`evidence/precision-paths-20261028/binary-path-audit.json`（位于任务外的同一项目 evidence，SHA-256 `20fe745573fcc63c7fd03d6bcea849f21f588dae89280c95de33f1ded5b28b0e`）记录一个安装时 `DispatchFFNCombine_17df546b46e1ea56007584fac7e6f429.o` 的 SHA `4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9`、`MMAD.s8` 静态反汇编命中和该对象 metadata 的 `tilingKey=1000010`。这证明该保存的安装对象包含这种指令和元数据；当前 generic CSV Name、48 条任务或已有 EP8 DB 的 `compute_name` 均未把具体历史 replay 绑定到该对象、key 或每次动态 tile。`blockNum=24`、`mixBlockNum=48` 也不能恢复内部 M/N/K、有效专家行或硬算力。故不能据此把观测时长和该对象的指令数/分块作为同一运行的正式性能界。下一步仅在存在带可信执行实例、完整 kernel symbol/文件 SHA、tiling 数据的既存文件时做有界关联；否则保持 UNKNOWN，不扩大重复扫描。本次只读既有材料。

### 专家行数据：已有目录取证范围

本机复核 `evidence/profiler-export-inventory.json`：它列出了 16 个 rank 的 `ASCEND_PROFILER_OUTPUT` 文件，但不是对 `PROF_.../device_*` 原始目录及其他 dump 根的全量清单。EP8 已核查 DB 表/字段名和固定短字符串的结果，按 `request-dynamic-metadata-coverage.md` 仅覆盖单库和限定术语；原有服务日志 tail 检索也不覆盖全部匿名日志。这些记录没有提供能关联本次 layer0/replay 的逐专家有效行数，不能由此宣称任何其他文件都不含该信息。

准备并通过 ZCode 执行了 `query_mc2_saved_dump_inventory.py`，SHA-256 `b2c6df7126ff56f0c54a4ec90df1bf815a634a53c0911162e430df7d9435fb53`：只从固定 manifest 确定 EP0/EP8 两个 rank 的 profile 根、export 根、device 根与 7 个明确子目录，读取文件名哈希/类型/限定标签和已 hash 的 profiler_info 中白名单布尔设置；不读取 dump、tensor、权重或请求 payload，不访问 NPU。清单里的文件名也不是内容或 replay 绑定证明；缺失不代表其他目录、其他 rank 或未列出的 payload 没有数据。

### stage180：两个已知 profile 原始目录的文件名盘点

本机 ZCode 记录 `operator-ceiling169-v2-180-mc2-inventory` 的退出码为 0，verified 显示 `model_verified=true`、观察模型均为 `deepseek-flash`。本机重算并与回执一致：结果 `mc2-saved-dump-inventory.json` SHA-256 `44d4652eed1e1d24cc33d9aa393f91f81b280071b4d8e1d05c7e33bff6ef99a8`；查询脚本 SHA-256 `b2c6df7126ff56f0c54a4ec90df1bf815a634a53c0911162e430df7d9435fb53`、wrapper `09bf0b05abf56d76541b2ee474d66f90793bbcf1101c5f887c5494db1e1ae7f9`，都与结果内记录一致。固定 manifest 的 SHA 为 `cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a`；查询同时按 manifest 校验了两份 `profiler_info` 的哈希。

EP0 与 EP8 分别检查 profile 根、`ASCEND_PROFILER_OUTPUT`、`device_0`/`device_8` 根和 7 个固定子目录。两侧各有一个已读 `device/data`，分别包含 46、40 个文件，名称未命中脚本限定的 `expert_token_nums`、`token_per_expert`、`cumsum`、`active_mask`、`dump` 或 `tensor` 标签；预设的 `dump`、`dumps`、`tensor`、`tensors`、`tensor_dump`、`dump_data` 六个 device 子目录不存在。其他三个已检查根目录的文件名同样无上述标签。文件名以 SHA 输出，未读取文件内容；原始 data 文件多数扩展名不在白名单，不能据此判定其内容类型。两份已校验 `profiler_info` 中，脚本遍历到的 `record_shapes`、`profile_memory`、`with_stack` 布尔字段均为 false；这只是这些配置字段的值，**不能推断 profile 不含形状或其他数据**，已有 CSV 确实记录了形状。

该结果没有定位到可直接以文件名识别的专家计数入口，也没有读取任何 raw data 或 tensor；不能证明已列出全部路径、其他 rank/目录或现有 raw 文件的内容不包含专家数据，更不能判断它能否绑定该层的具体 replay。因无来源契约的明确入口，不继续对 raw 文件做关键词泛扫；保留动态专家行数 UNKNOWN。整个执行仅检查已有目录与 profiler_info，不读取请求文本、权重 payload 或设备。

### stage249：EP0 / EP8 analysis.db 的 schema 与通信矩阵边界

对现有 profile 索引内的 EP0、EP8 两个 `ASCEND_PROFILER_OUTPUT/analysis.db` 只读检查表名和列名，没有读取数据行。两库各有四张表 `CommAnalyzerBandwidth`、`CommAnalyzerMatrix`、`CommAnalyzerTime`、`StepTraceTime`；没有表或列明确命名为逐专家计数、QLI 有效长度或 QLI metadata。表与列列表均完整、无截断；此结论仅指**命名和 schema**，不能推断字段内没有相关内容、其他数据库或文件没有对应数据，也不能证明采集时缺少这些数据。两库大小与既有索引一致，查询前后内容 SHA 一致、未见 WAL/SHM；本次观察的 EP0 / EP8 SHA 分别为 `a3c1b8526e209acf4479785fdcb1a385ccd48f86c224f31914a114e414680ffe` / `5ad9ac17680dc96a5e8403e55022688ac9f25dd75ed1ac5b9900b163b0ed189a`。索引原无内容哈希，故它们只是本次读取指纹，不能称为采集时冻结哈希。

其中 `CommAnalyzerMatrix` 有 `src_rank`、`dst_rank`、`transit_size`、`transit_time`、`bandwidth`、`step`、`group_name`、`hccl_op_name`、`op_name` 等列；这份 schema 不提供已验证的 MC2 图 replay 唯一标识。保存的 MindStudio 字段说明 `system_tuning.md:2561–2578` 描述按迭代/通信域的通信矩阵，`Total Op info` 是所选范围内算子统计和，分类 `total` 是同类总传输量除总传输时长；这说明通信矩阵**可能包含汇总口径**，并不确证当前 DB 表每个字段的生成器、单位、聚合粒度和版本。先前 stage144 在 EP0/EP1 的通信 JSON 矩阵中检查的是六条**草稿 all-to-all** 名称，未匹配，不能说它已经检查过本次 EP0/EP8 `analysis.db` 的全部行或首层 MC2。

现有第 0 层 `DispatchFFNCombine` 是融合 TASK，已有 `COMMUNICATION_TASK_INFO` 的精确关联未闭合；stage167–169 的 attr / group 哈希也不足以绑定同次 collective、特定 replay 或该表行。先前 analyzer 源码还显示可能按名称与组聚合，且本次 DB/JSON 生成链和来源覆盖未证。因此现有字段即使包含非零 source/destination 与传输数字，也无法直接把它算作指定第 0 层单次 MC2 的通信量、等待或瓶颈。**本次未读 Matrix 行或据字段名解读其数值**；若将来有同版本 exporter 映射、可信实例键及 replay 的明确绑定，才可设计定界查询。没有这些前提，重复按名称或同 step 查聚合值不会解决本任务的归因缺口。

schema 输出 `mc2-qli-analysis-schema.json` SHA-256 `639667ce7ac192284bfd93e06987c97f576b50415b7918fbe37aa6f0c65f4430`；查询 `query_mc2_qli_analysis_schema.py` SHA-256 `7a580f3fa734366eb71f0cbf1846edb9b4bf804026db07e0b831fadc65347f30`、wrapper `run_mc2_qli_analysis_schema.py` SHA-256 `0704fd26984c78165d4adb02db30748945fd65fef8da071f80e16586716348aa`。来源说明快照 SHA-256 `4c431dac256f6bb69927b721d49316f2716e2b1fc2228924d8d8a84bff5d9fa3`；相关既有矩阵查询和 analyzer 边界见 `DRAFT-EP7-QUERY-METADATA.md` stage141–144。`/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-249-analysis-schema.exit-code` 为 0，配套 `.verified.json` 为 `model_verified=true`、模型 `deepseek-flash`。无新 NPU 工作。
## 诊断服务配置日志的只读查询校验记录

- stage252 查询在挂载验证时失败，未读取日志。脚本要求 staging 将任务根目录直接挂载为同名 `/work/...` 子目录，而现存创建脚本 `start_target.py:59–60` 记录的是 `/data/tiankuan/wio/main` → `/work`；报错是泛化的 mount 校验失败，单凭该错误不能排除诊断挂载也不匹配。
- stage253 修正 staging 映射后，在读取之后的前后 `inspect` 元组比较处失败，结果未输出或保存。该比较检查 ID、image 和完整 mounts 列表；没有比较容器 state。仅凭这一失败不能确定发生了哪种变化。
- stage254 只读两次检查固定 staging/诊断容器，退出码 0，ZCode `model_verified=true`。两次 ID、image、日志路径相关绑定不变；两容器完整挂载列表的原顺序哈希均改变，而按 `(Destination, Source, Type)` 排序后的完整对象哈希均相同。该结果说明本次检查中出现顺序不稳定，是 stage253 报错的**相容原因**，但没有保留 stage253 的两份原始元数据，不能证明那一次唯一原因或排除当时其他变化。stage254 不读服务日志、不操作设备。

已准备 v3 查询脚本供审，按**完整对象的 canonical JSON** 排序后比较所有原始 mount 字段，以避免同三字段键重复时的排序歧义，同时保留容器 ID、image、精确 bind、两端 nested overmount、文件路径、大小和读前后状态校验；输出依旧仅限白名单配置值、行号和全文哈希。v3 尚未执行；不能因 stage254 的证据声称诊断配置已取得或历史执行分支已确定。

stage255 随后执行 v3，ZCode `model_verified=true`、退出码 0；保存 `agents/diagnosis/diagnostic-runtime-config-evidence.json`，固定诊断实例的日志 SHA-256 `51375755909e205512f1a7781627d79cca628d5970b198a51ab9de7247ee295d`。第 24 行按白名单匹配 `enable_fused_mc2=1`、`enable_flashcomm1=true`、`enable_shared_expert_dp=true`。stage256 对相同哈希的第 24 行进行有限类别检查，`agents/diagnosis/diagnostic-line24-context.json` 记录 `api_server` 与 `non_default_args`，退出码 0、`model_verified=true`；因此三值是诊断 APIServer 报告的非默认请求参数，仍不能视作 worker resolved 配置。其他目标字段及固定告警本次未匹配，不表示 false 或告警不存在于未覆盖的表达形式。此前“v3 尚未执行”仅描述 stage254 当时状态；更新记录没有额外读日志或运行 NPU。

### stage257：首层逐卡范围与 40 层内排序

复用上述 16 rank × 3 个内部采样序号、40 层的 MC2 归属结果：每张卡每轮各有 40 条，同模型、主 stream、对应层窗口包含、物理输入首维为 3；1920 条映射和首层 48 条身份校验见 `module-priority-evidence.md`“40 层对照”。每轮 DP0、DP1 各有 8 张卡，比较的是**每张卡独立算出的任务持续时间**，不是跨卡时间戳相减。下表首层范围单位为 μs；后两列是在各卡各轮中，首层是否严格长于该卡其余 39 层，及首层在 40 个时长中按降序的名次范围（名次 = 1 + 严格更长的其他层数）。

| 采样序号 | DP0 首层范围 | DP1 首层范围 | DP0：首层严格最长 / 8，名次范围 | DP1：首层严格最长 / 8，名次范围 |
|---|---:|---:|---:|---:|
| 1 | 270.500–274.395 | 2027.879–2035.220 | 0/8，15–19 | 8/8，1 |
| 2 | 388.333–395.332 | 938.121–944.321 | 0/8，3 | 8/8，1 |
| 3 | 349.080–356.853 | 1658.007–1663.267 | 0/8，4–5 | 8/8，1 |

每轮 DP1 的首层最小值均大于 DP0 的最大值；而且在 DP1 的 24 个卡内组中，首层均严格长于该卡同轮的其他 39 层。因而在这批记录中，首层组间差异并非只由少数慢卡拉动组中位数造成，DP1 首层在本卡内突出也不是跨卡合并排名的产物。第 0 层仅是 **MC2 任务时长**的比较，不能扩展为总层窗时长、芯片利用率或端到端瓶颈排名。

三个内部序号来自一次 profile，不是独立实验；对应 ordinal 与历史图映射沿用前述归属条件。组间序号相同并不证明相同请求、相同输入/动态专家负载、同一 collective 或跨卡同步；物理 M 一致也不等于实际工作量一致。比较持续时间不提供跨卡先后关系，也不能确定通信、调度或其他因果，更不能将组间差值称为可恢复时间、可实现提速或理论性能界。

只读复算：`python3 agents/diagnosis/recheck_layer0_mc2_strict_order.py`；脚本 SHA-256 `9c10409bb5d1ed95f054b4f74caf3c87a59c447d40a44921e0484944c5484a5b`，结果 `layer0-mc2-strict-order.json` SHA-256 `5458b86389c62304f0401e0905f9a5f7b8f3bd1b8282496aa66a34d1618e0211`。脚本固定先前 40 层校验脚本与结果的 SHA，并核验其列出的输入 SHA；复用原脚本的逐任务身份、层窗和物理形状校验。ZCode stage257 退出码 0、`model_verified=true`、观测模型 `deepseek-flash`；对其工具输出解析出的 JSON 与本机结果逐项核对相等。该步骤仅用已保存数据，未启动 NPU。

### 首层窗口按 MC2 前、中、后分解

沿用 stage257 的 48 个已归属首层窗口，逐 rank、逐内部采样序号使用该卡的整数纳秒端点，验证 `层窗起点 ≤ MC2 起点 ≤ MC2 终点 ≤ 层窗终点`，并逐窗严格计算 `首层窗时长 = MC2 前段 + MC2 持续时间 + MC2 后段`。前段中位数此前已有记录；这里补充后段、范围及同窗完整分解。每组各 8 张卡，表中每格为最小值 / 中位数 / 最大值，单位 μs：

| 轮次 | 组别 | MC2 前段 | MC2 持续时间 | MC2 后段 | 首层窗 |
|---|---|---|---|---|---|
| 1 | DP0 | 374.4920 / 377.8460 / 378.8530 | 270.5000 / 273.2445 / 274.3950 | 86.8980 / 89.1185 / 92.4180 | 735.9450 / 740.1755 / 742.8650 |
| 1 | DP1 | 379.3400 / 383.7720 / 387.0720 | 2027.8790 / 2031.1890 / 2035.2200 | 88.2600 / 89.9485 / 94.6380 | 2502.6600 / 2505.1195 / 2509.0900 |
| 2 | DP0 | 391.4520 / 394.7225 / 399.0120 | 388.3330 / 392.1725 / 395.3320 | 87.3580 / 88.7280 / 93.4180 | 873.7600 / 875.5425 / 880.7630 |
| 2 | DP1 | 388.2000 / 392.2520 / 396.6920 | 938.1210 / 942.9210 / 944.3210 | 88.4980 / 90.4590 / 94.4380 | 1421.8200 / 1424.2015 / 1427.4910 |
| 3 | DP0 | 383.6330 / 386.5820 / 390.9720 | 349.0800 / 353.9730 / 356.8530 | 86.9380 / 88.2980 / 94.2990 | 825.7000 / 829.4735 / 836.2040 |
| 3 | DP1 | 387.0720 / 390.5220 / 393.2320 | 1658.0070 / 1661.8765 / 1663.2670 | 87.7000 / 90.3685 / 94.1580 | 2137.9600 / 2141.8475 / 2146.4370 |

在这些已选窗口里，前段及后段两组的范围接近；每轮组间的大幅 MC2 任务时长差在首层窗口范围内清晰可见。三段相加等式只对**单张卡的单个窗口**成立；分别取出的中位数不必来自同一张卡，不能相加解释窗口中位数，也不能把两个组中位数的差说成已识别的可优化耗时。

前后段是局部主流层窗的时间跨度，并非单独算子、芯片空闲或等待；其中与其他流的重叠及未导出的活动未知。首层 MC2 是融合任务，无法从这个分解判断其内部通信、计算或等待各占多少。跨组同序号未证明同请求、相同专家负载、同次 collective 或跨卡同步；本次对比不提供跨卡因果或可恢复收益。三轮也非独立重复实验。

离线复算：`python3 agents/diagnosis/recheck_layer0_mc2_window_decomposition.py`；脚本 SHA-256 `888d79713c6af721f50872ac9b1cdead15ab42543058eec64f438e13752504e8`，含 48 条逐卡数据和分组统计的 `layer0-mc2-window-decomposition.json` SHA-256 `c4d766d66fe71a3d22349f1053f2567ad707d79ce94741b99ecc45125f03f371`。脚本固定并校验既有 40 层归属脚本和结果哈希，并复用其对输入哈希、各任务身份和完整包含关系的核验；另逐行核对 timeline 中 MC2 起止与已归属首层任务的起止完全一致。不进行跨设备时钟相减、不访问 NPU。

**stage258 复算与旧表对照：** 本机 ZCode `operator-ceiling169-v2-258-layer0-breakdown` 退出码为 0，`model_verified=true`、观测模型为 `deepseek-flash`。独立解析其工具输出，与保存的 `layer0-mc2-window-decomposition.json` 逐项比较相等。按原精度核对：每轮每组的 MC2 中位数与 `40layer-mc2-dp-comparison.json` 的首层结果一致（差异小于 `10⁻⁹ μs`）；以前记录的三个组间时长中位数比值仍分别为 7.434、2.404、4.695（显示三位小数）。将每组的 24 条样本合并后，其层窗和 MC2 中位数与 `layer0-dp-existing.json` 中对应 pooled 统计也一致（同样小于 `10⁻⁹ μs`）。既有“第 0 层到 MC2”表的前段和 MC2 数值与本表一致，旧表显示三位小数，本表保留中位数的半纳秒，因此显示末位可能相差 0.0005 μs；两处浮点小数格式化可能选取相邻显示值。

这些对照确认的是同批已选记录的统计与复现，不引入新的独立样本。三段的各组中位数不能直接相加，MC2 内部的计算、通信和等待依旧未拆解；不能从中认定跨卡同请求、同步或因果。本次复算只使用既有数据，未运行 NPU。

## 二进制身份的现有采集入口

已索引安装对象包含 `MMAD.s8` 和 `tilingKey=1000010`，而已查采样 CSV 名称仍为泛化 `DispatchFFNCombine`，选定 EP8 profiler 数据库字段没有二进制 SHA、实际 kernel symbol 或每次 replay 的 tiling；反复相同 task ID 也不能建立唯一 replay 对象。[现存 MindStudio 文档](../../sources/primary/msinsight/system_tuning.md) 的 1837–1853 行描述另一种 `cluster_analysis.db` / `OperatorMFU` 产品可显示设备 kernel 名称与时间，但已保存的 profiler 导出索引未列出该文件。这仅是限定索引结果，不证明其他路径不存在，也未验证历史材料可生成该产物、名称唯一性或导出版本一致。若后续找到实际产物，仍须证明与本次 task/replay 和已加载 ELF 的映射，才可将内核内部源码推断绑定观测。

## EP8 PMU 精度候选与历史 op_summary 文件范围

复用 `mc2-pmu-replay-links-ep8.json` 的7个同 globalTaskId TASK 与7组 PMU，TASK 与 `rank_target_calls.csv` 以同卡 start/end、model、stream、task 各唯一匹配；PMU 取结束时刻最近且相差不超过21ns的组，每组唯一。globalTaskId 在7次调用复用，时间近邻只是候选，不是已经验证的 replay 键。若这7对确属同一次事件，用 Decimal 比较保存 PMU 与 CSV 三位值：AIC MAC 和 MTE2 按0.001最近 half-up 舍入均为7/7相符，按向零截断分别为3/7、5/7相符。由此在该候选关联成立的条件下，全部采用简单截断的解释与部分值不符；不能证明配对、全局舍入方式、物理采样精度或 AIV 行为。数据库 float 转存多位也不证明硬件精度；既有按最近舍入的敏感性区间仍是条件性分析。

复算脚本 `recheck_mc2_pmu_candidate_rounding.py` SHA 33ba48cd72bada4700ab021b5ee39c3c9fd2b9f593eee7bd5e4963c7b29bd234，输入 PMU SHA ef5d4b8400d2615de23e57c92b882861d0db78badd40b2cab9c28948e776e966、CSV 摘要 SHA 19ff5196289b93ee53bcb4a6a27946dbff8c84ebd28988fd40b4fe0bacefbe17；结果 [mc2-pmu-candidate-rounding.json](mc2-pmu-candidate-rounding.json) SHA 930aab0f35a77607be801ef847cad8effd462f732893cf7aee26bc452742d9a0，ZCode stage273 exit 0、model_verified=true 且输出 SHA 一致，仅本机 CPU。

根据当前 parser 的文件查找规则，对已有 EP0/EP8 固定 PROF 根及各自 mindstudio_profiler_output、analyze 三个顶层目录只读检查目标文件名：两个根各2项，未发现匹配 op_summary CSV；四个子目录不存在。范围只覆盖两个 profiler 和所选目录/文件名，不能证明别处或其他命名没有历史文件，也不等于未采集。身份通过已保存 profiler_info、start_info SHA 校验。ZCode stage274 在本地源码哈希抄写错误处中止且未远程查询，改正后 stage275 exit 0、model_verified=true；[文件清点](ep0-ep8-op-summary-inventory.json) SHA 56e8a57d28e59bcd00a2a7edd3c9c9f295ee4c6e3affba8da95039073a37bba4。没有启动 NPU 或修改推理代码。

### 同总行数不能识别专家分布：条件反例

对于 24 个本地专家和容量至少 48 的条件示例，`[48,0,…,0]` 与 `[2,2,…,2]` 均有 48 个有效行，但分别有 1 和 24 个非空专家。冻结 `dispatch_ffn_combine_kernel.hpp` 的 GMM1（454–497 行）和 GMM2（561–611 行）按专家构建截断后的 `currentM`，分别更新 `blockScheduler`；这表明单个总行数不足以恢复各组输入形状。反例按容量 48 对源码截断规则进行整数复算：`python3 agents/diagnosis/mc2_distribution_counterexample.py`；脚本 SHA-256 `e4ea8601a8e0b99feb2f7b1c330ebd697bad32ba9d12b6fbb81ac709f3bccf00`，冻结 kernel 源 SHA-256 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`。这只是数学输入例，未证明这两种向量在历史调用可达，不证明历史采用该源码对象、各组实际任务数、时延或利用率。至少需要保存被正确同步、被下一次覆盖前的完整计数向量和同次容量，并绑定 rank、层、replay 与具体 task；单有计数仍不能确定跨 rank 等待。

本轮仅对 stage180 已检查的 EP0/EP8 同一固定 `device/data` 目录打印文件名和大小，其内容分为 `aicpu`、`ffts_profile`、`stars_soc`、`lpmInfoConv` 四类及对应 `.done` 文件。已有 `existing-mc2-pc-capture-audit.json` 记录 16 rank 相同名称类别，故不视为新增的数据源；未解码文件内容，不能排除其中或其他路径有可绑定数据。本轮未找到新的实际计数、mask 或路由值入口，未访问设备或启动负载。
