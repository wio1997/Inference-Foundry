# 硬件地图：拓扑、条件性能与竞争关系

用途：为 GLM-5.3 W8A8C8 / vLLM-Ascend 的执行方案提供可检验的资源依据。当前研究设备为 32、33，用户已恢复设备操作并授权停止这两台原实验占用；正式目标保持 PD 分离。2026-10-17 已完成首轮只读身份/拓扑/软件/资产核查，已有实测与未知项如下。已补初始条件通信计时和 HCCL SDMA 运行日志；33完整模型单机容量/短请求已验证，真实 PD KV 路径仍待验证。后端按用户最新要求优先核实灵渠MemCache，暂停新增Mooncake实验。

地图要可追溯、可复核、能暴露错误。硬件身份、当前软件能发挥的性能、特定负载下的竞争分别记录；实测值不是物理常数，测得的最高带宽也不是所有实现的性能上界。未知项保持未知，不填零或用宣传值代替。新证据与地图冲突时重新核验，不让旧地图直接否定新方案。

## 2026-10-17 首轮现场快照

原始证据与 SHA 见 [inventory 摘要](evidence/20261017-inventory/summary.json) 和每 run 的 context/result/stdout；资源变更以后续 [HANDOFF](HANDOFF.md) 及执行凭据为准，不能沿用初始预留状态。

| 项目 | 两台实测 | 适用限制 |
| --- | --- | --- |
| 组成 | 每台 8 张物理卡、16 die、每 die npu-smi 显示 65536 MiB；逻辑 ID=Phy-ID | torch 可用总量不等于物理标称，33 die0 见 65787658240 bytes；其余 die 的运行时容量待测 |
| 同卡/异卡 | 成对 die 为 SIO；异卡为 HCCS_SW | 来自设备拓扑，不是带宽结果；不同 HCCS_SW 对不预设等速 |
| 超节点 | 两台采样 die0 均 superPodID=1，size384，server index=31/32；采样 HCCS health=OK | 不证明跨机已能传或与更多节点等速，未用 size384 推断拥有全部资源 |
| CPU/RAM | Kunpeng 920 7280Z，4×80 core×2 SMT=640 逻辑 CPU，8 NUMA，约 2 TiB RAM | NPU PCI sysfs NUMA=-1，亲和未知；不改系统 NUMA 信息 |
| 宿主网 | enp162s0f0/f1 报 100000 Mbps、NUMA5；enp194s0f0/f1 报 25000 Mbps、NUMA4；bond4 报 50000 Mbps | sysfs 链路值，不是有效吞吐；管理地址不是 NPU 数据路径 |
| NPU 网络 | 16 die 普通 IP getter 均返回未配置；RoCE link=DOWN / health=Init；vNIC=UP | 不据此宣称 HCCS/SDMA 不通；不擅自配置 RoCE |
| 软件身份 | driver25.5.2；镜像 CANN9.1.0、vLLM0.30.0+empty、Ascend0.30.0rc1、torch-npu2.10.0.post4；MemCache/MemFabric均1.2.0 | 完整 imageID/两仓库 SHA 在 HANDOFF；16rank实际V1加载日志已核验；quant method/cache tensor仍待运行对象证据。灵渠网络版本未知 |
| 权重 | 两台 196 文件/764240832702 bytes，已有全量校验完成凭据；本线清单身份与当前尺寸核对通过 | 复用凭据，没有重复全量 payload 哈希；运行时 W8A8C8/C8 尚待确认 |
| 干扰窗口 | 两台 5 秒样本 NVMe 读为0、写<0.01 MB/s，管理网有小流量，已知传输 worker 未见 | 只是瞬时诊断，不能替后续运行保证隔离；共享 HCCS 其他节点仍可能竞争 |

| 物理卡 | die (逻辑/Phy-ID) | PCI BDF (die0 / die1) |
| --- | --- | --- |
| 0 | 0 / 1 | 0000:9D:00.0 / 0000:9F:00.0 |
| 1 | 2 / 3 | 0000:99:00.0 / 0000:9B:00.0 |
| 2 | 4 / 5 | 0000:95:00.0 / 0000:97:00.0 |
| 3 | 6 / 7 | 0000:91:00.0 / 0000:93:00.0 |
| 4 | 8 / 9 | 0000:8D:00.0 / 0000:8F:00.0 |
| 5 | 10 / 11 | 0000:89:00.0 / 0000:8B:00.0 |
| 6 | 12 / 13 | 0000:85:00.0 / 0000:87:00.0 |
| 7 | 14 / 15 | 0000:81:00.0 / 0000:83:00.0 |

## 已验证范围与初始计时

HCCL 诊断的完整归档扫描中，同卡与异卡各有6条 SDMA link success、50条 isUsedRdma[0]；跨节点每端各3条/25条，isUsedRdma[1]、isInterRdma[1]、forceRdma[1]均未见。每端完成15项×12次计时，接收/归约逐次校验通过。完整日志在相应目标节点及 Mac 已校验备份，仓库仅留归档身份、必要摘录及 sdma-analysis。

下表为 ERROR 日志级别下**接收端**区间的中位数，包含 Python/后端调用与设备完成同步。每项12次，属于条件参考，不是纯链路延迟、p99、PD延迟或物理上界；未采用 INFO 诊断时延。两台 NPU 当时由本任务预留，权重已传完；没有证明整个384规模互联独占。

| 路径/方向（物理 die） | 1 KiB (µs) | 64 MiB (µs) | 计时原始证据 |
| --- | ---: | ---: | --- |
| 33:0→1 (SIO) | 253.5 | 680.5 | [timing](evidence/20261017-inventory/node33-hccl-sio-sdma-timing-7060a227303a/metrics.json) |
| 33:1→0 (SIO) | 254.3 | 667.2 | 同上 |
| 33:0→2 (HCCS_SW) | 240.6 | 766.5 | [timing](evidence/20261017-inventory/node33-hccl-hccs-sdma-timing-99d5e7cdbfaa/metrics.json) |
| 33:2→0 (HCCS_SW) | 226.4 | 758.8 | 同上 |
| 32:2→33:2 | 253.2 | 748.3 | [接收端区间](evidence/20261017-inventory/node33-hccl-cross-sdma-timing-bb2e3d559af7/metrics.json) |
| 33:2→32:2 | 227.3 | 780.0 | [接收端区间](evidence/20261017-inventory/node32-hccl-cross-sdma-timing-bb2e3d559af7/metrics.json) |

每个完整 JSON 还保留64KiB/1MiB/16MiB和bf16 AllReduce、原始各rank ns数组及正确性。只比较这些已测条件，不从小消息区间直接认定全部开销属于 SDMA 引擎。

跨机组：rank0=32/物理2/SDID130285570，rank1=33/物理2/SDID134479874；容器用户编号均为0。全部32个die的原生spod-info都报告 superPodID1，32/33分别server index31/32；完整SDID映射见 [32](evidence/20261017-inventory/node32-sdid-efa2c7300796/stdout.json)、[33](evidence/20261017-inventory/node33-sdid-efa2c7300796/stdout.json)。本组未设置 HCCL_LOGIC_SUPERPOD_ID 或外部rank-table环境变量，采用HCCL root-info动态建域。此证据不排除集群其他域存在逻辑划分。

## 官方能力与现场差异：只保留改变实验的资料

用户描述“只有 SDMA 到超节点”按**本线数据通路约束及现场待核实信息**记录，不写成 910C 硬件仅支持 SDMA。SDMA 是搬运机制，HCCS 是互联链路；HCCL/MemFabric/MemCache 是库或后端，SSH/管理 IP 是控制网络。这些层次分别取证。

MemCache新增实证：镜像中MemCache/MemFabric均1.2.0。32[2]↔33[2]两客户端以`device_sdma`、每rank1GiB DRAM池、0HBM池、无SSD测试64KiB/1MiB双向写池/读池，全部接收SHA匹配。HYBM实际data-op=2与安装头文件SDMA位值一致；两台profiler各4个`SDMA_SQE`，与4个put/get区间逐一匹配。因此本次**合成数据路径**已验证为SDMA，流程为源HBM→DRAM池→对端HBM；不是模型PD或直传证明。清理阶段仍退出133/139，最小init/close-only也复现，Meta停止退出134；功能与生命周期不能合并记为通过。原件/派生分析见 `node{32,33}-memcache-sdma-close-profile-*`，续接状态见HANDOFF。灵渠网络版本>=1.5的配套条件尚未独立核实，不从路径成功反推出版本。

| 第一手资料支持的能力 | 现场已确认 | 待验证 | 改变的实验 |
| --- | --- | --- | --- |
| [CloudMatrix384 §3.3](https://arxiv.org/html/2506.12708v2#S3.SS3)：双 die 与多互联平面 | 8卡/16die，SIO/HCCS_SW，vNIC UP | 逻辑超节点划分、全 die SDID、跨机实际选路 | 用原生 getter+rank/SDID 对照，不套用论文带宽或规模 |
| 同文 §4.2.1：AIV-Direct 针对 SDMA 发起开销 | 当前小测走 AICPU/SDMA；小消息含明显接口完成成本 | 真实模型中发起/等待占比 | 保持全部 SDMA，不开发 kernel；先测消息合并、注册/缓冲复用与事件依赖 |
| 同文 §4.3.2：AIC/AIV/SDMA 分工和 microbatch 流水 | 本线尚无真实计算共跑证据 | HBM/链路/队列争用是否抵消重叠 | 用已有正确算子/接口比较独跑与相关共跑，总完成时间重新计量 |
| 同文 §4.3.3：KV 使用独立 RDMA 平面隔离 decode 通信 | 本线 SDMA 约束；未验证 KV backend | KV 与 TP/EP 是否共享瓶颈 | 不照搬隔离假设；验证 KV 分块/节流/错开发起与无策略对照 |
| [CANN9.1 HCCL 开关正文](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_INTER_HCCS_DISABLE.md)：A3 FALSE 为 HCCS/SDMA，TRUE 为 RoCE/RDMA | 已显式 FALSE、PCIE=1/ROCE=0、AI_CPU；同卡/异卡日志出现 SDMA link success、isUsedRdma[0] | 跨机、不同组、真实 PD connector、communicator 级覆盖 | 诊断日志与正式计时分开；HCCL 证明不外推至 KV |

论文条目为机制参考，现场列才是本线证据。预分配/双缓冲、按数据就绪发起、KV 分块/节流及错开 decode 关键通信都是候选，需由现有接口、真实消息量、正确性与总完成时间逐项接受或否定；不预设增加并发必然更快，也不预设 MoE 为首个瓶颈。

## SDMA 有界首轮

用户要求数据搬运全部 SDMA。先取得本任务资源，再核对实际 backend/任务选路；控制面 TCP 单列。初始 HCCL 配置显式 `HCCL_INTER_HCCS_DISABLE=FALSE`、`HCCL_INTRA_PCIE_ENABLE=1`、`HCCL_INTRA_ROCE_ENABLE=0`、`HCCL_OP_EXPANSION_MODE=AI_CPU`。A3 的选路与展开模式依据 [HCCS/SDMA 官方说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_INTER_HCCS_DISABLE.md) 和 [CANN9.1 展开模式](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md)。环境变量不代替实际任务证据；communicator 配置可能覆盖环境。

| 问题 | 最小覆盖 | 状态/判别 |
| --- | --- | --- |
| 同卡 vs 异卡 | 33 的 0↔1、0↔2，必要时 0↔8；32 做必要复核 | int8 send/recv 双向、bf16 all_reduce；1KiB/64KiB/1MiB/16MiB/64MiB；先为诊断 |
| 相关组 | 单机 TP4 组及拟用 EP16 组 | 基础 all_gather / all_reduce / all_to_all，组内分片和字节口径单列；不将两台默认合成32-rank服务组 |
| 节点间 | 32 die2 ↔ 33 die2，必要时对应多路 | HCCL 基础与实际 PD backend 分开，保留真实 SDMA 路由、两方向和端到端完成 |
| 竞争 | 两对独立 SDMA 与各自单跑，再按真实 KV/EP 消息补充 | 关注两侧减速与总完成时间，避免将重叠区间重复累计 |
| 真实 KV | actual connector 的注册/产生/传输/消费，int8 KV/indexer 元数据 | 单机容量和功能允许后运行真正 P→D，必须排除 D 重做 prefill |

初始脚本 [hccl_probe.py](tools/hccl_probe.py) 的计时为各 rank 独立 `perf_counter_ns` 包围通信调用+设备完成同步；barrier 完成、填充和逐次正确性验证在计时外。12 次样本不报告稳定 p99，也不当作纯链路时延、PD 服务时延或物理全局上界。日志/profiling 的诊断结果与无 profiling 测量分别保留。并发/大组/跨机入口按结果补充，不创建全排列搜索器。

## 地图包含什么

| 层次 | 最少记录 | 用途与边界 |
| --- | --- | --- |
| 组成与拓扑 | 宿主机、物理卡、die、设备标识、容器可见设备与 rank 的映射；CPU/NUMA、HBM、机内/跨机链路与共享资源；资源预留、进程及采样时间 | 管理 IP 不代表 NPU 数据通路；同超节点标签不证明任意两点等速。rank 映射和资源归属随部署核对。 |
| 条件性能 | 消息/算子类型、大小或 shape、dtype、布局、rank 组及分布、执行模式、软件身份；延迟与有效吞吐 | 测试相应条件下的可达成本。先覆盖关键范围，随实际消息与执行片段补齐，不穷举所有 shape。 |
| 竞争与状态 | 多路通信、计算与通信、HBM 访问同时发生时的成本；必要缓冲、状态驻留与运行峰值 | 分别测独立和并发执行的同一份工作；重叠的时间段不能直接算作节省，关注总完成时间及两侧减速。 |
| 软件与接口能力 | 实际算子/通信后端、量化和 KV 路径、分块粒度、异步完成语义、图模式及布局限制 | 框架未暴露、接口不支持与物理不可行分别标注。更换现有后端后相关性能结论可能变化。 |

## 首轮有限采集

1. 复用已有快照，核对 32/33 的身份、可用资源和路径。为物理连接与运行时协议分别留证据；发现冲突先定位来源，不从其他节点的镜像或健康状态推断本节点。
2. 不加载完整权重，先用必要缓冲区测机内与跨机点对点通信：少量消息量级、两个方向、单路与相关多路并发。初始范围是诊断条件；取得真实 KV/集合通信大小后再细化，记录算法切换等不连续行为。
3. 按拟用 P、D 各自的 TP/EP 组测相关集合通信。不能默认将两台全部 die 编为一个通信组。保留真实 rank 放置；按需补不均匀消息分布和 rank 到达偏差。HCCL Test 可用于对应版本的集合通信基础测试，不替代实际 PD 传输后端。
4. 核实真实 KV connector 的生产、传输和消费路径，再测从数据可发送到 D 可正确使用的交付成本。必要时分辨整理/拷贝、注册、排队、传输及完成通知；存在重叠时不能将分段耗时简单相加。检查实际 C8 存储/传输/消费，不从参数名或目录名判断。
5. 结合首个机制补计算通信竞争、HBM 或 CPU 提交成本。使用相关的真实执行片段；无需先测遍算子。记录额外缓冲与 KV 容量的交换，不能把孤立通信峰值当作服务运行时能力。

前两项通常不需要完整模型，但仍是设备操作，按 [AGENTS](AGENTS.md) 的现场边界执行。实际 PD 路径及内存容量需要适用的软件和运行验证；基础通信正常不能证明 1P1D 可用。各项按当前决定选择，不要求先建完整地图再探索机制。

## 每个结果如何留下证据

- **身份与条件**：设备/rank 映射、软件提交/镜像及相关配置、时间、消息/shape、并发、其他作业与影响结果的运行状态。区分设备/官方资料、用户提供、历史快照、本线实测和推断；有冲突的条目保留冲突。
- **计时与正确性**：确认操作完成及接收端数据正确，不只量异步提交返回。记录预热、冷初始化、测量区间、重复数和波动；跨机时间戳须说明对时与误差，优先使用同一时钟的端到端区间。短样本不包装为稳定尾延迟。
- **字节口径**：说明有效载荷、每 rank/总量、单向/双向、GB/s 或 GiB/s；集合通信的算法带宽与换算总线带宽分开，不能直接当物理链路实测流量。避免重复累计并发区间。
- **原始依据**：保留可重复命令、脚本版本、原始结果位置与摘要。使用真实发生的计时；诊断 profiling 与无 profiling 对照，必要时交叉核对设备/后端记录，避免测量工具改变结论。
- **适用与失效**：写明覆盖范围与关键未知。软件、拓扑、rank 布局、传输协议、执行模式或竞争条件改变时，重核依赖这些条件的条目，不全量机械重跑。预测与测量分开，用额外点检查预测误差。

## 怎样支持研究

每个候选说明它依赖哪些地图条目，以及预计减少的工作/搬运/等待和新增的计算/驻留/竞争。沿候选的合法依赖重新判断请求关键路径与持续服务容量；原框架的同步边界不自动成为必要依赖。

两机证据只覆盖实测关系。扩到更多节点时，检查通信对端数、每 rank 形状、共享链路与内存余量如何变化，对新增竞争做目标规模短测；不能按卡数线性外推。缺地图的区域可提出假设与廉价验证，但不能据此宣称收益或不可行。

首版只需让首轮部署与机制判断有依据：拓扑可核验，关键成本有条件和波动范围，影响下一决定的未知项可见。测量投入按当前问题限定，证据已足以区分方案就停止加密采样，出现异常再局部补测。结果随实验产生，暂不另建数据库、完整自动搜索器或庞大记录体系。
