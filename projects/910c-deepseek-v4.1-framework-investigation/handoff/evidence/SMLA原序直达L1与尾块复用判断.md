# SMLA 原序直达 L1 与尾块复用判断

介绍：沿当前 selected IDs/可见长度 → 原 V0 物理 gather → 三槽 GM panel → Cube QK/PV 的真实 L1/LoadData/MMAD 读者，判断整段能少搬什么、能共享什么。目标保持每个替换原语逐位一致，包括原 BF16/FP32 舍入及合法回退。数学上共用 KV 不允许改变 softmax、PV 的 K 归约顺序或首末 partial 的发布语义。

到了哪里：11 份当前实际源身份及限定 ranges，见 [审阅表](smla-direct-l1-source-review.json)。6 个原 V0 函数原样抽取后，在本机编译，333 个边界 fixture、46,184 个已写行地址与独立计划一致。另用两份旧原生 QLI 回执重建并校验 prepared-ID hash，配两种 PA 页表情景得到 12 例；检查 1,239,552 个 QK/PV 活跃 NZ 坐标的源地址，以及安装头文件实际 2201/3510 字段分支。本轮没有 NPU 原语数值、真实 DMA、服务或新 profile 实验。

结果：**大方案待确认，新增已确认可实现原语 0**。原顺序计划有 CPU 地址证据，尾 128 行驻 L1 有源码容量与功能任务寿命见证；实际 ND2NZ padding、LoadData/MMAD、异步 flags/预取代价仍缺逐位验收。已停止四个具体配方，不扩大为全 SMLA 否定。全链路未扫全，11 个 SHA/ranges 也不构成全部合法分支已扫证明。

## 原顺序与硬件限制

`prepare_indexer_indices` 实际先筛 `0 <= id < (position+1)//ratio`，再按位置升序排序并写 `-1` 后缀；lengths 输出在小规模融合，大规模另做 sum。它目前没有 KV 页表输入，不能直接产出物理地址，也不能替每个真实读者证明 cache/page/causal-cap 代次。

V0 并非顺着逻辑 ID 原样写 panel。快 pair 按物理地址升序写；原 byte stride 太大、负值或边界触发时保留 pair 逻辑顺序、逐条回退。两个 AIV 各自压紧自己的半区，在 frozen 512 项时从 0 和 256 开始，不跨半区合并；正 ID 被当前 reader cap 裁掉时，旧 panel 的洞可能仍保留旧值。新计划需保存原 destination ordinal 和 64 位源地址；存在活跃洞的 packet 保留旧路径。333 个用例中，朴素逻辑顺序有 104 例不符，7 例触发活跃洞回退。

安装 CANN 的 910C/2201 `Nd2NzParams.srcDValue` 是 **uint16、单位元素**。D512 BF16 的物理行距 127 为 65024，128 为 65536；原 struct 在本机实际编译后后者变 0。3510 分支才是 uint64。原 V0 的 int32 byte-gap 配对范围更宽，不能把其 stride 原样搬到 ND2NZ。大 base address 保留 64 位；新 stride 不可表示的 pair 拆成单行，**仍保持原 emitted 顺序**。这不否定大地址单行直读。

原 common 中的 `DataCopyPABySlots` 在当前 arch22 没有调用者，只按逻辑 row 写，既不做原快 pair 排序也不做原半区压紧。直接替换的配方停止；不能把“函数已存在”算作当前服务已有安全直读。

## QK/PV 布局、复用与等待

原 L1 分配为 QP 四块共 256KiB、KV 三块共 192KiB，合计 448KiB。QK 是每个最多 128 行、两个 D256 NZ tile；PV 每个 D128 strip 的 work slot 则拼两个 128×128 NZ 子块，B 的 LoadData3D 转置，K 依原顺序逐 128 累加。不能把 QK 整块作为 PV 的单个 work tile 直接传入。

**较大联合方向**：保留最后一个最多 128 行 N tile 的两个 D256 QK 块，占两块 KV L1；用第三块作 work。实际 preload 为本次 `QK(i)` 然后 `PV(i-1)`，不是同次 QK 后立即 PV。若上一任务是 CMP，先来的下一任务 ORI QK 需使用 work，才轮到上一 CMP 的 PV；其前面的 K 块仍从源读，最后 128 行从两块 pinned QK tile 按四个 D128 strip 读回，保持原 K 顺序。1/2/4/8 ORI tasks/query 的功能级 owner/release 见证及活跃地址映射通过。相邻 CMP、特殊 M split、tail、FD/HEAD_RATIO_ONE 等未资格化分支需要原路径回退。

这仍不是异步实现：必须重写或验证 mte21KVIds 的取得/释放、MTE2→MTE1 完成、crosscore flag3 与三环退役。两块 pin 会减弱原三槽预取，代价不能漏记。K512 全部 KV 是 512KiB，连同原 QP 256KiB 不可能放入原 448KiB；停止“原容量里全驻留”这个配方，不否定改变 QP 分配/算法的未来更大方案。

位门槛仍缺 ND2NZ 非 16 对齐 group 起点、N padding 写入和尾块、实际 LoadData3D 转置、MMAD/Fixpipe 全原语输出，以及 NaN/次正规数/零/无有效 ID 与 fallback。CPU 活跃地址相同只说明任意输入 BF16 bits 从同一处取，不能证明硬件 padding、数值或并发。不能提前 PV 或重排 K，不能删原概率舍入和 softmax 归约。

12B/item 的保守表示可放 64 位 base 与 32 位描述：512 内原目的行 9 位、run length≤128 用 8 位、正行距≤127 用 7 位，余 8 位保留 flags。每行仍可保留一条记录，合并只影响搬运；计划构造/页表访问并未免费。共享 key 至少包含 index/KV source pair、request/query-mask 域和 cache/page-table generation，图稳定 bank 到所有读者末读才复用。

## 匹配旧数据与净价

[价格原件](smla-direct-l1-price.json) 精确复现上一轮 54 个含计划成本情景，再算 108 个联合条件与 20 个指令成本敏感性。仍用原 C8 2560 窗、1840 native、30 typed 组及固定 EP 到达区间；给全 TP 最有利信用和流量差额 50% 暴露。速率 2405.303756 GB/s 来自同旧 cohort 五种 byte-count 的 BF16 连续 TensorMove 一份 payload 中位速率乘二。它**不是 sparse GM→L1 带宽或指令发射速度**。

| 条件：K512、payload 参考速率 75%、h5µs、plan factor1 | 最有利请求的条件收益 | 所有请求最小条件收益 |
| --- | ---: | ---: |
| 不保留尾块 | 1.395453% | 0.641482% |
| 50% 尾块合资格 | 1.530869% | 0.705455% |
| 100% 尾块合资格 | 1.666284% | 0.769428% |

这不是 P90、预计值、测得净收益或全请求下界。后一条件的逐窗净差仍有负数，范围 −4.3146 至 183.9789µs，不裁掉负值。最有利请求到 1% 所需 payload 速率：无 pin 为参考的 63.4239%/71.6694%，全 pin 为 55.4959%/62.7108%（plan factor1/8），仍需另付未包含的指令成本。

窄 fixture 不支持一概按随机最坏情形淘汰：12 个旧 QLI prepared-ID × PA 情景为 **24–78 条 direct ND2NZ/查询**，保留尾块后 **20–62**；旧 dense 为 24。随机全 pair K512 可达 direct1536/retained1280，全 single 可达3072/2560。不能用这些合成情景替代真实 ID/页分布。

在额外指令按 `ceil(queryRows/24)×extraInstructions×ns` 付价的**合成均衡 24 AIC 模型**中，retained62、每条额外5ns时最有利条件 1.6060%；retained1280、1ns时1.2708%，5ns则所有请求条件转负；retained2560、1ns时最有利也只有0.8714%。ns、负载均衡与有效带宽都未测，这组数用于指出会改变去留的缺口，不能据此停掉真实低指令分支。

另有已测旧 **stock GM dictionary prepacking**：Q416/G64/D512/K512、两种 ratio、U1024/8192、15轮 AB/BA，warm 或 256MiB touch 后，构造已预付。30 调用条件差约0.0093–0.0687ms，对照旧1%参考4.488ms，个别 ratio 为负。停止该原配方/fixture 的独立开发；它仍保留原 V0/panel/原 dense-to-L1，并未测试本轮 direct+tail。旧 whole-output bit observation 仅属于旧 fixture/包，不能替新原语验收。

## 地图处置

| 分类 | 项目 | 下一步或停止边界 |
| --- | --- | --- |
| 待确认 | 原序 plan + direct L1 + 尾块复用 + 原首段交付的大范围 | 缺真实 K/原序 affine groups/page epoch、ND2NZ padding 与完整 NPU 原语位门槛、新 flags/预取和付价。只补会改变去留的参数；不先开服务整跑。 |
| 已停止 | 逻辑序 `DataCopyPABySlots` 直接替 V0 | 104 用例地址反例；改变原顺序即不准入。 |
| 已停止 | 用 A5 宽 stride 或原 V0 byte-gap 驱动 910C ND2NZ | 实际字段宽度与单位反例；合法拆分的新方案不在停止范围。 |
| 已停止 | 原七块 L1 中全部驻留 K512 KV+QP | 容量不符；两块尾 pin 是另一个联合候选。 |
| 已停止 | 旧 stock GM prepacking 的原配方/原1%独立预算 | 已测低值；不外推 direct/新布局/六普通。 |
| 已确认可实现 | 本轮新增 0 | CPU 地址/struct/坐标 proof 不升级为 NPU primitive bit PASS。 |

下一段转到 **Engram 紧凑 scale 的 checkpoint → host/UVA 注册 owner → device pointer bank → lookup/dequant → 末读/close/reload**。它已有18,432,548,400 B 条件容量差和编码255反例后的CPU穷举，但真实成对加载/注册/退役链尚未闭合。先保留 SMLA 上列具体缺口继续扫其它交付，不为同一个缺口无限等待。

证据：[前沿](smla-direct-l1-frontier.json)、[探索记录](smla-direct-l1-exploration-notes.json)、[原 V0 CPU](smla-transfer-plan-cpu-a2/result.json)、[字段/NZ CPU](smla-l1-layout-cpu-a1/result.json)。a1 的 mock 编译失败保留；frontier 初稿 indexer range 写到310越过实际EOF298，被断言拒绝，修正为259–298，未扩大覆盖。原 finished loops 不修改。

环境：171已按统一入口核对；仅 owned 容器源码/安装头文件只读与本机CPU编译/旧数据复算；无本轮 NPU/模型/推理请求/新profile/服务更改/显式远端写入。HANDOFF 中另一个已授权的配平 profile 控制器及其文件不受本扫描操作，不能声称整个服务器没有并行工作。
