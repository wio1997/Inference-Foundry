# MC2 异常层的 AIV 读流水：与源码同步轮询的交叉核查

## 官网 512 B 地址规则与专家计数轮询的静态关系（2026-10-29）

[昇腾 CANN 9.1 开发指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90/%E8%8E%B7%E5%8F%96%E6%80%A7%E8%83%BD%E6%95%B0%E6%8D%AE.md)针对所列 Atlas A2/A3 等产品描述：MTE2、MTE3、Scalar 访问 GM 时以 512 B 粒度处理，同时请求落在同一连续 512 B 范围可能因一致性而串行。该指南版本、产品定义、历史 kernel 和 shmem 窗口映射未绑定本次设备，不能将这个通用机制当成已观察到的 MC2 冲突。

对保存的参考源码，在 `sizeof(int)=sizeof(int32_t)=4`、有效索引、地址运算不溢出等条件下，令 `E=expertPerRank`，`P=128×ceil((EP×E+1)/128)`，`B=shmem()+offsetPeerTokenPerExpert`。`dispatch_ffn_combine_kernel.hpp:247–249,694–701`、`utils/layout3d.hpp:5–19`、`utils/const_args.hpp:6` 和 `moe_v2_common.h:48` 给出第 d 个来源、该来源第 j 个同步检查的起始地址 `B+4×d×P+512×j`，其中 `0≤j<P/128`。不同 `(d,j)` 的起始地址至少相距 512 字节，即使 B 不对齐也不会落在相同的 512 字节对齐区间。仅作假设示例，E=24、EP=8 对应 P=256、每来源 2 个起点及 1024 B 行距；EP=16 对应 P=512、每来源 4 个起点及 2048 B 行距，均不是历史运行实际参数证明。

这是很窄的地址起点结论：`utils/hccl_shmem.hpp:66–80` 的循环每次请求读取 8 个 int32，即 32 B，而非单个起点字节；B 的 512 B 对齐未证，若 B 在块内偏移超过 480 B，读取范围可能跨入下一 512 B 区间。peer 发布写与轮询/其他读取也可能触及相同区域；不能由起点不同断言所有并发访问不会竞争。实际调用、地址、并行、规则是否适用于 shmem 映射及历史二进制均未知，因此不能排除读写竞争、归因低活动或给出耗时。发送循环使用 `coreIdx` 指定 peer，不能无条件用循环变量推断远端目标。

来源 SHA-256：kernel `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`、layout `54f6e597229c16caa367c6e690a8f7d985c5b9217537e0ec702e910e4115f66f`、shmem `596bc8a0c28748aade2e5f2e978f18a6e593d146a8cf07fe4aeac8d36c19bad6`、const `4376995bf93267caf13e7fc373299f526a3c145833f6abb1b2f7b6e9caa8a3d7`、moe common `7406a0d5abaace9e8268f688fe7d217cda4ff88ee399d46bcf6084a1699b92ec`。只读源码、官网与既存资料，未启动 NPU。

**结论：提出一个可区分的待检假说，尚未找到历史任务内部阶段或根因。** 对已有同 digest `1b826883...`、EP8–15 的层 14 和层 39 共 16 条已匹配 DispatchFFNCombine，不启动设备，直接比较相同导出字段。每个 rank 的两层 shape、dtype、format、core 和 block 字段均相同，但动态路由、有效行、内部状态和历史二进制身份并未绑定。

| EP | task duration 14÷39 | L14 字段值÷1000 | L39 字段值÷1000 | MTE2 时间字段 14÷39 | MTE3 时间字段 14÷39 |
|---:|---:|---:|---:|---:|---:|
| 8 | 19.344 | 20.103 | 0.512 | 39.244 | 0.328 |
| 9 | 19.488 | 20.052 | 0.222 | 90.484 | 0.835 |
| 10 | 19.498 | 20.418 | 0.262 | 77.797 | 2.850 |
| 11 | 19.510 | 20.212 | 0.252 | 80.289 | 1.700 |
| 12 | 19.499 | 20.154 | 0.281 | 71.708 | 2.035 |
| 13 | 19.516 | 20.204 | 0.234 | 86.436 | 2.319 |
| 14 | 19.479 | 20.021 | 0.309 | 64.733 | 0.697 |
| 15 | 19.483 | 19.959 | 0.265 | 75.325 | 0.453 |

表里字段原名为 `aiv_mte2_time(us)`、`aiv_mte3_time(us)`；中间两列只是原始数值除以 1000，不代表已验证的实际毫秒耗时。MTE2、MTE3 不是已验证的搬运字节、带宽或各阶段墙钟。不同流水/核心可重叠，单位和统计归约的具体版本语义未闭合；采集开启 profiler 且两层实际工作量未证明相同。这个对照只说明显示值的比值具有显著不对称性，**不证明读带宽瓶颈、20 ms 等待、真实吞吐或可兑现收益**。

冻结参考源码 `dispatch_ffn_combine_kernel.hpp:811` 调用 `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2`；`:693–703` 的参考实现对 peer expert-count 的状态槽在 `gm_signal_wait_until_ne(sync_check, 0)` 后才读取计数。`utils/hccl_shmem.hpp:66–82` 中该 helper 在循环中对状态地址执行 `DataCopy(ub, sig, 8)`，设置并等待 `MTE2_S` 事件，直至本地结果不等于指定值。因此，**若**历史任务确实执行此参考源码所编译的相应分支，等待生产者更新状态也可能表现为多次 MTE2 读取；另外，`:846–866` 的 peer payload 搬运同样包含读取。该源码与此次历史 kernel 对象尚未绑定，动态调用、迭代次数和统计口径均未知。不能从代码存在推断本次一定执行、耗时或对端落后；也不能从高 MTE2 证明在轮询、排除数据搬运，或由此判断其他 rank 是原因。`:749–788` 的另一个 `UpdateAicFlags` 含读循环但本源码树内未发现调用，不能拿它证明本次执行。

**区分证据：**在可证明属于同一 kernel 对象、该次调用和同 core 的设备流水/PC 或稀疏阶段标记中，区分标志循环读和 peer payload 读，并同时绑定 expert count、有效行、各记录覆盖与丢失；核对实际版本 MTE2 字段如何计数。若只能取得聚合字段，应保留两种解释与未覆盖数据，不使用同窗接近来归因。现有资源限制下不新增设备采样，继续查历史数据是否含这些可绑定的记录。

**复算**：任务根目录 `python3 agents/diagnosis/compare_mc2_aiv_read_poll.py`。输入 `diagnostic-20261029/mc2-extra-pipe-fields-16-tasks.json` SHA-256 `57d21f3ad1789ec38fdf0499ea65b6f5ca793e676d68193d3dbdc320b96f755c`，其中保存各 rank 原始 CSV 的 SHA、完整 task 身份和全字段；见 `RECAPTURE-MC2-NOREPLAY-EVIDENCE-20261029.md` 的原始提取与约束。脚本校验 16 条数据、两层输出和 shape 等同后仅输出计算，不写结果、不读取 prompt 或权重。

## 已保存采集中的指令与 PC 能力核查

对该次 16 rank 采集逐个读取设备侧 `sample.json`：16/16 的 `pcSampling` 为 `off`、`instrProfiling` 为空，`ai_core_profiling_mode` 与 `aiv_profiling_mode` 均为 `task-based`。这分别是保存的配置字段及其值，不把空字符串擅自解释为所有潜在内部能力的语义关闭。逐 rank 的路径、SHA-256、原始设备数据文件名前缀与这四项值见 `agents/diagnosis/existing-mc2-pc-capture-audit.json`。

EP8 的既存 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db` 只读检查得到 28 张表及所有列；其中 `TASK_PMU_INFO` 有 task ID、计数器名称和值及时间戳，`TASK` 有任务时间及身份；表和字段均未列出可供对应 AIV 指令源码的 PC 或指令轨道。所有原始设备文件的**文件名前缀**仅发现 aicpu、ffts_profile、stars_soc、lpmInfoConv、ts_track 五类，不能以名称代替对二进制内容的完全解码。

因此，在这批已有导出、明确配置与检查过的目录范围内，**没有找到能把 L14 的 MTE2 读绑定到标志轮询或 payload 的 PC/指令级轨道**。这不是证明 raw 绝对没有任何隐藏信息，也不是其他时间的采集都不存在；更不能证明轮询或搬运任一解释。继续逐次尝试用现有任务级字段拆阶段缺少依据；如果未来资源约束发生变化，需要先验证工具支持、真实二进制与 task 身份，再评估采集覆盖，不能按默认 replay 运行多 rank 内核。本次没有新设备工作。

复核：在任务根目录执行 `python3 agents/diagnosis/audit_existing_mc2_pc_capture.py`；脚本通过自有 stage 容器只读原有 16 个配置及 EP8 SQLite schema，不读取张量、权重或 prompt。结果 JSON SHA-256 `8363012053e686d5e06303194d039b3394f5f038796311b25ba77a03a323dfb6`。

### EP8 两条目标任务的精确绑定

同一 EP8 DB 里，只用 task ID、stream ID 47 和 model ID 4294967295 查询时，第 14 层的 task 44217 与第 39 层的 task 45363 **各有两条候选 TASK**。加上此前通过 scope→flow→设备任务→CSV 绑定的精确开始时间（以十进制 μs 精确乘以 1000 转成 ns）后，各只剩一条：第 14 层 `globalTaskId=2538`，第 39 层 `globalTaskId=3506`。脚本查询表达为：

```sql
SELECT globalTaskId,startNs,endNs FROM TASK
WHERE taskId=? AND streamId=? AND modelId=?;
SELECT p.timestampNs,s.value,p.value FROM TASK_PMU_INFO p
LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTaskId=?
ORDER BY s.value;
```

对候选列表再严格筛选 `startNs == CSV 开始时间×1000`；非唯一时脚本报错。两条任务各取得 28 行 PMU，**每条任务的 28 行都只有一个 timestamp**，且各是所匹配 TASK `endNs` 前 20 ns。其计数器名为 AIC/AIV 的 total cycles/time 和流水 ratio/time 等；这支持这些行在所查数据库中是每任务一组汇总，而非所需的逐阶段指令时间线，**不扩大到其他任务或所有原始采集记录**。DB 原字段数值比 CSV 显示精度更高，例如两条任务 AIC MAC ratio 分别为 0.002591239876063173 和 0.16437222164807858；高精度仍不能补足计数器语义、工作量与归因缺口。

复算脚本与 JSON 为上一节所列文件；JSON `ep8_targets` 保留输入条件、粗匹配数量、精确匹配的全局 ID、每行指标、去重时间和文件输入哈希。


## 同一源码分支的 payload 逻辑工作量（2026-10-29）

保存的 `dispatch_ffn_combine.h:216–227` 将 AType 指为 row-major int8；kernel 文件的 `:76,313–355,842–862` 把 AType 用作 peer 数据类型、把 `ubMoveNum` 设 2，并以 `DataCopy(buf, src[inputOffset], rowNum*(hiddenSize+UB_ALIGN))` 请求 GM→UB 连续读取。另一个已存同版本头文件 `utils/const_args.hpp:9` 定义 UB_ALIGN=32（SHA-256 `4376995bf93267caf13e7fc373299f526a3c145833f6abb1b2f7b6e9caa8a3d7`）；同快照 `dispatch_ffn_combine.h:42–43` 包含该常量和 kernel，HCCL_COMM 分支按 `utils/const_args.hpp` 包含，非 HCCL_COMM 的搜索路径未单独确认。[昇腾 CANN 9.1 GM/UB 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/data_move_guide/gm_and_ub_data_movement.md)指出这个连续 DataCopy 形式的 count 以元素为单位；文档版本与历史 9.2 仍未绑定。

令 `r_i` 为**每一次实际进入该 helper 调用**时，经过 `maxOutputSize` 截断后的非负行数，`R=sum(r_i)`。若该源码、常量、int8 模板、K=5120 和接口语义都适用于所讨论的调用，则 helper 中连续读取请求的逻辑元素数为 `R*(5120+32)`，即逻辑字节数 `5152*R`；请求次数为 `sum(ceil(r_i/2))`。尾批只按实际剩余行请求；若 R=0，此 helper 的读取循环不执行。行包含宽度 K 的量化数据及预留区，不能把额外 32 字节全算作有效 payload。公式仅统计此 helper 的读请求，不包括 routing、轮询、计数、写出、其他阶段，也不能推断远端网络流量、HBM 事务字节、命中率或 MTE2 时间；peer 可包含本地对象。

仅从 R 可以得到条件性的调用次数范围 `ceil(R/2) <= calls <= R`；例如 R=4，若四行在一个 helper 调用中，2 次读取；若分成四个各一行的 helper 调用，4 次读取。两者逻辑请求均为 20608 字节。这是数学构造，不代表真实专家分布或硬件性能；要计算精确次数仍需逐次/逐 peer×expert 的截断行数，即使知道总 R 也不够。这指出了为什么相同物理 M 和相同行数仍可能有不同请求组织，但**不能把已观测的长短差归因为它**。

当前实际 R、各次行数、历史二进制/tiling 和运行分支未绑定；本节不是对指定历史调用的实际工作量测量，不收紧全模型或算子的硬性能上界。它为之后如果出现同次绑定的有效行数据提供可复算入口，现有 task 级 PMU 无法反推出这些行数或轮询次数。本轮只读源码、官网与现有元数据，不启动 NPU。
