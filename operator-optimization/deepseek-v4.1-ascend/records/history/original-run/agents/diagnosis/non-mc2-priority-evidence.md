# MC2 之外的排查优先级：已有 trace

对 `captured-layer-attribution/operator_metrics.csv` 的内部轮次，按 operator type 合并 16 rank × 3 轮 × 40 target 层中已归属的任务。脚本 `python3 agents/diagnosis/rank_non_mc2_operators.py` 可复算，输入哈希、完整结果与限制在 `non-mc2-operator-aggregate.json`。同类内可能有不同层与形状；时间之和包含同时运行的 rank／流，**不能当端到端时间或收益**。MAC 数据是 AIC 周期加权代理，不能当整卡利用率或等待占比。

| 已归属任务类型 | 调用 | 采样耗时之和 | 单次平均 | AIC MAC 代理 |
|---|---:|---:|---:|---:|
| DispatchFFNCombine（参照） | 1920 | 613.750 ms | 319.66 μs | 6.96% |
| QuantBatchMatmulV3 | 6144 | 177.236 ms | 28.85 μs | 9.91% |
| HcPre | 3840 | 135.207 ms | 35.21 μs | 5.74% |
| TransposeBatchMatMul | 1920 | 100.969 ms | 52.59 μs | 10.00% |
| VllmAscendQuantLightningIndexerV2 | 384 | 81.936 ms | 213.37 μs | 4.56% |
| SparseFlashMla | 1920 | 70.378 ms | 36.66 μs | 3.12% |

**优先候选：Indexer。**在这批已归属样本中，它既有较长的平均单次任务时间，也有较低的 AIC MAC 活动代理；采样耗时总和排名第 6。其调用比每层出现的算子少，不能只看单次时长。需要进一步按层、形状、输入长度和 rank 分组，厘清实际被等待的路径；低 MAC 本身不足以说明计算、访存或同步中谁造成慢。

**Q 投影是另一个候选。**独立的精确形状样本（16 rank，1920 次，M=3）中位数 42 μs，MAC 代理约 10.6%，MTE2 代理约 72.9%，见 `evidence/q-projection-16rank-diagnosis.json`。上表 QuantBatchMatmulV3 是多种矩阵算子和形状的混合集合，不能与 Q 投影逐次一一对应，也不能用 MTE2 代理断定 HBM 带宽饱和。其频繁调用提示后续应按形状及依赖而非类型名称继续拆分。

**Attention 有低活动指标，但影响未确定。**SparseFlashMla 代理 3.12%，单次平均 36.66 μs；它是 AIC 流水指标，不能说明整个 attention 低效。比较前须分解其关联 AIV、形状和有无 indexer 的不同层，确认与其他任务是否重叠。

**Engram 与调度先补关联。**已有 16 rank Engram 精确命名内核样本中，hash 中位数约 127.13 μs、host-UVA lookup 约 12.16 μs，跨形状／阶段混合且不代表完整模块；只读样本中 lookup 和 all-gather 直接数据依赖未证明。更正：此前跨 DP 图提交和第 0 层开始时间比较缺少跨设备时钟误差上界及同次 replay 身份，不能作为上游先后或等待的证据，也不能据此归因 scheduler 或 Engram。`engram-kernel-profile/summary.json`、`evidence/engram-gather-candidate-example.json` 和 `evidence/graph-submit-diagnosis.json` 留有证据。

下一个不使用 NPU 的实质动作：对 Indexer 做按层、rank 的源表分布，确认每层与相关 attention 窗口的相对时间；核查同流时间顺序及已证明的数据依赖。只有形状、请求和同步关联足够时，才能考虑它在端到端上是否暴露。

## 已归属 target 量化矩阵乘 / HcPre 的物理分组（stage202）

`matmul-hcpre-shapes.json`（SHA-256 `2d09b0226e8882039d9b866848ab3a9b7625b66f67787b3eb44cf81d0439028e`）使用 `query_matmul_hcpre_shapes.py` 及固定哈希的 16 份导出 CSV、窗口和 manifest；逐 rank 自身时钟筛 model 49 及主流、3 轮 × 40 层完全落窗的这两种 Type。查询无问题（`COMPLETE`、issues 为空）。本地独立核对：QuantBatchMatmulV3 四组共 6144 行、任务时长求和 177236.450 µs；HcPre 一组共 3840 行、135207.268 µs，与已有 `captured-layer-attribution/operator_metrics.csv` 的同类型对应层汇总次数及微秒总和分别精确一致。该交叉检查不保证查询以外全部 kernel 或其他流的采集完整，也不等于端到端覆盖。

| 导出类型与输出物理形状 | 调用数 | 单次中位数 μs | 类型内任务时长总和占比 | 层覆盖 |
|---|---:|---:|---:|---|
| QuantBatchMatmulV3 → BF16 [3,32768] | 1920 | 42.000 | 45.870% | 每层每卡每轮 1 次 |
| QuantBatchMatmulV3 → INT32 [3,4608] | 1920 | 25.600 | 28.024% | 每层每卡每轮 1 次 |
| QuantBatchMatmulV3 → BF16 [3,5120] | 1920 | 21.279 | 23.089% | 每层每卡每轮 1 次 |
| QuantBatchMatmulV3 → BF16 [3,4096] | 384 | 13.800 | 3.017% | 仅第 2、8、14、20、24、28、32、36 层，每卡每轮 1 次 |
| HcPre → BF16 [3,5120] + 三个 FP32 输出 | 3840 | 34.819 | 同类型 100% | 每层每卡每轮 2 次 |

四组 QuantBatchMatmulV3 的前两个输入均导出为 INT8 / INT8 且权重格式为 FRACTAL_NZ；输出 [3,4608] 这一组输入记录只有 3 项、输出 INT32，其他三组均为 4 项输入及 BF16 输出。物理签名说明调用输入输出及导出 metadata 差异，不足以推断量化参数语义、实际计算路径或最终模型层身份。虽然表中按跨 rank、跨轮任务总时长占比可看出混合集合的组成，**绝不能将该占比用于墙钟时间、端到端收益或不同组可加速百分比**。

此前 `evidence/q-projection-16rank-diagnosis.json` 的 Q 形状关联样本也是 1920 条、16 rank、相同内部窗口、M=3、中位 42.000 µs；本次 [3,1280]→[3,32768] 组与其高度相容，**不作为新增 Q 投影候选或额外收益重复记账**。统计条件及计数一致不是对每条原始任务身份的逐条交集证明。对于 [3,4096] 的特殊八层，已存 `F05-F09-SHARED-QR-QUANT-REVIEW.md` 记录与 indexer Q 源码维度相容的局部例子；仍缺 tensor/storage ID 与逐调用 Python 绑定，不能仅凭此次形状和层覆盖断言身份或归因。第三组的 [3,5120] 输出及第二组的 INT32 输出目前同样未证明具体角色。

HcPre 虽然物理元数据仅一组，但不能把两次调用归为相同作用；`agents/fusion/F03-HC-SOURCE-CONTRACT-AND-TILING.md` 的 stage122 已通过同流 marker 次序和结构不变量作 attention/FFN pre 的条件归属，且已分析按 token/feature 分块及外部 pre_mix 的重组候选。本次不能代替 tensor 身份或 binary/source 绑定，也不重复申报 F03。该采样的 HC 当前结果只为分组范围和可追溯性提供旁证。

下一步：若历史 profiler 有有效的调用连接信息，优先针对 INT32 [3,4608] 与 BF16 [3,5120] 各挑选一个既有完整层窗口，沿导出任务、source 调用接口和上下游依赖匹配，寻找不同角色和可合并条件；不能仅凭维度指认。若现有连接不足，明确为未知，转向其他已有证据充分的优化机会。无新增设备负载或目标代码修改。

## SparseFlashMla：AIC / AIV 与标量流水对照（只读复算）

复用固定的 16 rank × 3 内部轮次 × 40 target 层窗口，逐行验 16 份 kernel CSV、窗口和导出 manifest 的 SHA-256；在每卡本地 model 49 主流、完整落窗且正时长的 `SparseFlashMla` 中，每个窗口恰匹配 1 条，共 **1,920** 条。两种正周期以及下列字段在这 1,920 条中均有效。`attention_engine_counter_by_layer.py` 可重算，`attention-engine-counter-by-layer.json` 保存输入哈希、各层各轮结果及 6 种导出形状。输出形状均为 `[3,64,512]`；输入的容器形状有 6 组，这些不代表实际有效长度、请求或 KV 访问工作量一致。

整批按各自引擎正周期分别加权，AIC MAC 代理 **3.117%**、AIC scalar 代理 **57.347%**；AIV vector 代理 **2.775%**、AIV scalar 代理 **53.030%**。已存 [system_tuning.md](../../sources/primary/msinsight/system_tuning.md) 将 scalar 描述为负责控制流和标量运算的单元，ratio 描述为相应耗时比例，但该文档与本次 exporter 的精确版本绑定未证；这些代理不是整卡利用率，**不同流水可重叠，不可相加或把余数算成等待**。这组结果排除“采样中所有所列流水导出比例都接近零”的狭义说法，不证明 CPU 调度、标量控制、通信、访存或某条指令是瓶颈。

举同一第 24 层：第 1、2、3 轮 16 rank 的任务时长中位数分别 **16.540、44.079、44.279 μs**；对应 AIC MAC 周期加权代理 **0.14%、4.80%、4.79%**，AIV vector 代理 **1.02%、3.83%、3.81%**；AIC scalar 代理 **55.96%、53.93%、53.87%**。中位数与加权结果各自汇总，未必代表同一张卡。跨轮实际输入和动态有效长度未知，不能将时长变化归因于某个计数器、路径或可实现优化。该采样经历 profiler，不能拿它直接替换无 profiler 的正式基线。

下一步若历史 trace 仍有可定位的同次有效长度或局部同步证据，可以对这些候选做同工作量条件下比较；目前这里的 MAC/AIV 数值和持续时间仍不足以确认资源瓶颈、空闲或收益。注意力计算、标量、访存可能互相重叠；计数器不能直接相减得到等待。

## INT32 4608 形状：已有活动指标样例及绑定缺口（2026-10-30）

本地既存 `evidence/trace-kernel-inspection.json`（SHA-256 `60302ecbef9e9b5fc15962212c763de460135ce067bbf2117ba9bb8275e27b15`）的 `cube_low[5]` 保存一条 `QuantBatchMatmulV3` 摘要：输入 `INT8 [3,5120]; INT8 [144,320,16,32]; BF16 [4608]`，与 stage202 的目标输入签名相容；该条导出 duration `25.959 µs`、`aic_mac_ratio=0.108`、`aic_mte2_ratio=0.6`，另有 `cube_utilization(%)=87.691`。它只保存了名称、输入形状/类型及这些字段，**没有保存该条输出形状、rank、layer、窗口、task、stream、起止时间、源 CSV 哈希、总周期和 host flow**。因此目前不能把它绑定到上述 1920 条集合、EP8 的 task1300 或具体模块，不能按这单条数值代表该组或做周期加权统计；采样来源及 exporter 合同也待闭合。

这三个字段分别来自导出的不同指标，不能当作可直接比较的整芯片利用率或空闲率，也不能以 MTE2 比例认定 HBM 饱和。另一个已核实的 CANN 解析示例表明 `cube_utilization` 与 MAC ratio 不等价，但示例属于不同任务，不能作为本行的来源证明。`layer0-dp-type-counters.json` 按 type 聚合三种矩阵形状，不能将其 AIC 代理或全部时间套到 4608 组。基线任务具体位置、25.980 µs 单例及共享专家条件候选见 [F22](../fusion/F22-SHARED-EXPERT-REVIEW.md)；其相邻任务的张量身份和 host stack 未绑定。现有 2026-10-29 的 M=595/643/1023 重采集关联不代表同一次 M=3 基线。严格硬容量、有效行、数学语义和窗口初态不足，未形成对实际样本的数值性能界，既有条件模型见 [冷态 GEMM 边界](../bounds/cold_gemm_bound.md)。

本次只读小型已有 JSON 并核对哈希，没有访问远程或设备。若后续能读取同一基线的已冻结 EP8 CSV / trace，优先用 rank、model、stream、task 和精确起止唯一定位 task1300，查询对应周期与 ratio，再核查 trace flow 是否匹配且有调用栈；时间邻近或同名不能替代对象与重放绑定。

### EP8 单样本精确绑定与计数器（2026-10-30）

随后通过只读 169 专用容器内**同一历史采集**，对固定 SHA 的 EP8 `kernel_details.csv` 和 `ascend_pytorch_profiler_8.db` 核查 model 49、stream 89、task 1300 与精确起止；CSV 和 DB 各唯一命中该 task。对应 INT8 `[3,5120]`、FRACTAL_NZ INT8 `[144,320,16,32]`、BF16 `[4608]` 到 INT32 `[3,4608]`，CSV 记录耗时 **25.980 µs**、`aic_total_cycles=955675`、`aic_mac_ratio=0.112`、`aic_mte2_ratio=0.614`、`aic_scalar_ratio=0.441` 和 `cube_utilization(%)=85.151`。这是确定属于原选定 EP8、layer24、run2 的单条导出记录，不能推广到全组；前三个比例对应所报字段，不能相加、当作整设备利用率或等待占比，也不能凭 MTE2 指标判断 HBM 带宽已饱和。

DB 该任务的 globalTaskId 19484 在 TASK 表共出现 **7** 次；该记录 connectionId 184330 在查询的 CANN_API 和 CONNECTION_IDS 表内没有匹配。这个查询没有涵盖全部可能关联机制，故不能断言没有 host flow；它也未将采样绑定到特定 shared-expert Python 对象、scale、NZ 地址或最终有效工作量。上节只有输入相容的 25.959 µs 样本与此任务不同，不应混成一个样本。

复现入口 `python3 agents/diagnosis/query_ep8_int32_4608_exact.py`，在持有已冻结数据的专用容器内执行；脚本只读两个固定文件、核对原有 SHA、查询一个精确任务并输出脱敏字段，不加载模型或初始化设备。结果 [ep8-int32-4608-exact.json](ep8-int32-4608-exact.json)，脚本 SHA-256 `201f07d46f7b7bc8112ebc0bcc53f699476283bb054189e4b830211517f6281f`。后续若另有独立可绑定的图节点或张量元数据再沿其追查；重复按同一 connection 查询不会增加身份信息。

### 原始 trace 中的同轨 flow 端点

按同一历史 EP8 trace 的精确任务开始时间定向解析全部 704,318 条 trace 记录，定位到一条与 task 1300 完全同 PID、TID 和时间的 `f` 端点，ID `791691321671680`（即 `184330<<32`）。此前已有 [flow 汇总](ep8-selected-flow-endpoints.json) 对该轮相同 ID 在预选窗口记录 **581 个端点**且输出前 30 条、存在截断；这个复用 ID 不能唯一配对本 task 的主机调用、层对象或 tensor。`is_torch_to_npu=false` 是当前脚本仅按名字/类别字面检查 `torch_to_npu` 的结果，不说明不存在任何 host-to-device flow；先前汇总将部分同 ID 端点类别标为 `HostToDevice`。这是比 DB 直连空结果更具体的关联线索，但仍不能确定实际 Python 算子调用或运行分支。

证据 [ep8-int32-task-flow.json](ep8-int32-task-flow.json) 与 `inspect_ep8_int32_task_flow.py`；脚本 SHA-256 `ae79293551722bfe2e2058cd05846cf3043e4c64681e134d6a9d195c325f7e71`。查询核对 trace SHA 与先前记录一致，该 hash 并非独立冻结认证；只查精确时间及所列字段，不能排除其他编码关联、时间未对齐的端点或原始其他工件。未执行设备任务。

## INT32 4608 组：1920 个任务的 AIC 计数器范围（2026-10-30）

使用同一冻结 stage202 的窗口、16 个 CSV 和完整签名（含名称文本的哈希）只读筛选：16 rank × 3 个内部轮次 × 40 层、model 49、主 stream、任务完全包含在对应窗口中，每窗**恰好一条**，共 1920 条。窗口、manifest、stage202 形状结果及全部 16 份 CSV 的 SHA-256 均与已保存值一致。没有把这些窗口当成独立实验，也没有证明相同有效行或工作量。该查询绑定采样任务的导出签名，仍无法仅凭签名确定它是 shared expert 或绑定 Python 对象、tensor 存储及历史二进制。

以 CSV `aic_total_cycles` 为正且对应 ratio 为 0–1 有限数为统计条件；三项均 1920/1920 有效、周期非正或缺失 0 条、正周期但 ratio 缺失或无效 0 条，因而三者本次使用同一个周期分母 **1,838,621,927**。下表中每项周期加权值均为 `Σ(cycles × ratio)/Σ(cycles)`；范围和中位数则是任务报告 ratio 的非加权统计。

| 范围 | 任务 | 时长 μs 最小 / 中位 / 最大 | MAC ratio 中位 / 加权 | MTE2 ratio 中位 / 加权 | scalar ratio 中位 / 加权 |
|---|---:|---:|---:|---:|---:|
| 全部 | 1920 | 23.040 / 25.600 / 32.739 | 0.112 / 0.11128 | 0.620 / 0.61438 | 0.462 / 0.46293 |
| DP0 | 960 | 23.040 / 25.4895 / 32.120 | 0.112 / 0.11170 | 0.623 / 0.61659 | 0.466 / 0.46707 |
| DP1 | 960 | 23.100 / 25.7095 / 32.739 | 0.111 / 0.11087 | 0.617 / 0.61217 | 0.458 / 0.45883 |

全组单任务报告范围：MAC 0.092–0.123、MTE2 0.517–0.664、scalar 0.378–0.539。先在每个 rank 内用同法计算：16 个 rank 各 120 条，MAC 加权值范围 0.10807–0.11443，MTE2 为 0.59670–0.62973，scalar 为 0.43304–0.48720；各 rank 的任务时长中位数范围 25.270–26.3195 µs。每个 DP×内部轮次各 320 条，时长中位数分别为 DP0 25.4195 / 25.460 / 25.6095 µs，DP1 25.6895 / 25.7090 / 25.7490 µs。

这些结果比先前混合 4 种形状的 QuantBatchMatmulV3 汇总更具体，但仍是**被选择任务的导出活动比例**，不是整 NPU 利用率、有效 MAC 次数、物理搬运字节或端到端时间；MTE2 占比不能说明 HBM 带宽饱和，scalar 占比也不能单独证明控制流瓶颈。流水可能同时活跃，比例不可相加作为总利用率。统计时长来自 profiler 执行条件，跨 rank/轮不应相加为墙钟或收益；三轮同属一次 profile，不能当成独立实验。已观察物理首维不保证三行均有效，动态输入、缓存、调度、实际数学语义和计数器聚合等仍是未知。也不能把本组 25.6 µs 与另有强假设的理想模型相除作为已证实际效率或可达加速。

离线查询 [query_matmul4608_counters.py](query_matmul4608_counters.py) SHA-256 `68742a6315235a2ad51aa1282ebc55e25cd72c4ab9d3a70b0d4327ed1cd61b3a`；结果 [matmul4608-counters.json](matmul4608-counters.json) SHA-256 `85b44a60b19c443b93efa5050c9f7e1f6e668fca0748d1635b956f11840ad57d`，含 16 rank、各 DP 和轮次统计及 16 CSV 哈希。另以分 rank 的周期分母与加权值复核整体：16×120 与 6×320 的数量吻合，3 项分母求和及加权合并均与整体一致；这只检查统计计算，不验证底层采集完整性和硬件语义。ZCode 为生成查询已调用，但超过限制超时（退出码 124）且未生成脚本；查询由 GPT 编写并检查。仅在 stage 容器只读历史文件，未启动 NPU 负载。
