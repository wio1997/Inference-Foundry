# Astra High：Loop065 双层 Bound 与下一架构实验

独立只读审查，2026-09-26。依据 Run99、242、247–250、256、267、287、292 和当前源码；未操作服务/NPU，未修改运行源码。本文是 Sol 的决策输入，不作 KEEP 裁决。

## 1. 结论

- **Hardware/Resource 与 Scheduling-aware/Product 的数值上限仍 unknown，证据不支持“接近极限”。** 目前最大的未闭合项是必要工作分布/复制、真实并发容量、请求级有效工作与完整依赖路径，不能由单层 H003 或连续局部失败替代。
- Run292 已把“本层 hidden AllGather 与 local Q 能否真正并行”从假设变成设备事实；同配置 A1 是否改善共同前驱→join→后续 collective 仍须完成。一次带 profiler 的 TPS 不是收益判据。
- 新的离线复核发现：**Run287 的 16.65% slot-cycle 已 parked，活跃请求平均接受输出 4.139，全部槽位有效输出平均 3.422。** 这是区别于非owner副本的另一项全执行架构空间；不是 16.65% wall-time 可删。无需重新采集 acceptance census。
- 当前 H003 matched A1/B 应完成以回收已有实验信息；下一项扩大搜索空间的首选是**一个真实 c4 层入口的 request-owner 更新 compact A/B**，解决 96→16 完整请求行是否语义可行、成本是否真的减少。固定槽位 parked 工作与请求级流水线作为第二条架构候选保留，不先把动态 refill 混进同一个干预。

## 2. 必须保留的资源口径

| 项目 | 已知事实 | 最大未闭合项 |
|---|---|---|
| Target arithmetic | Run242 routed GMM 155.676 GFLOP/rank/cycle；Run256 Compressor 两投影 58.385 GFLOP | 均为当前调用的部分 canonical 算术；全 Target、DSpark、prefill、路由分布与跨rank复制尚未闭合 |
| HBM | Run247 整窗口中位 read 18.949755 GB / write 2.380327 GB；GMM 9.244 GB | 不是 compulsory bytes；packed weight 8.462 GB 来自不同 route 样本，1.092 不是同cycle可删除重读比例 |
| 非GMM | quant 3.465、Compressor 1.530、attention 1.062、other 3.618 GB 描述当前读量 | owner复制、typed KV/state历史读、layout/materialization、缓存驻留；不能相减得到冗余 |
| HCCL | Run250 每窗口 265 events / 25,651,200 reported payload bytes | 必要跨切分信息、物理link bytes、拓扑cut、rank arrival、Graph中attainable进度未知；payload不是wire量 |
| 并发能力 | Run292 AG/Q 真重叠；Run284 Compressor overlap 同时令 Rotary 膨胀 | AIC/AIV/HBM/HCCL共享资源的联合可行容量域未知，单算子最优值不能同时套用 |

给定一个合法算法/数据分布 a，物理资源松弛可写为：
L_resource(a) = max_r W_r_required(a) / C_r_upper；
再加拓扑cut的信息传输下界。只有容量**上界**才能构成严格耗时下界；测试观测到的带宽说明可达到，不能证明不能更快。

Scheduling-aware 松弛还需合法 DAG 的最长依赖路径，并满足资源、storage lifetime、事件与跨rank rendezvous：
L_schedule(a) >= max(L_resource(a), CP_semantic(a))。
这仍可能低估真实resource-constrained makespan；可实现工程方案须给出具体排程及并发实测。不同 a 的 work/communication 会改变，不能把当前265 collective或96行复制当作固定数学约束。Product 最终是有真实请求到达、prefill、逐请求 recurrence、输出提交的 makespan；不能把不同轨迹的平均 cycle 时长与理想 acceptance 随意相乘。

## 3. 哪些依赖必要，哪些只是现有边界

**高置信真实依赖：**

- 同一 token 的 attention输出→FFN输入→下一层 hidden：deepseek_v4.py:985–1002 的残差/HC路径。跨层重排必须等到所需的完整或合法partial结果，不能因允许架构变化删除数学 RAW。
- layer 内 local hidden→q_a→qr；qr同时可供应 q_b/main Q 与 indexer query。gathered hidden→wkv/indexer cache/main compressor。QLI需要其 query 和可见 indexer cache；Sparse需要 main Q、QLI选择、SWA/compressed可见缓存。不是“整层只能串行”。
- acceptance需要当前 Target 的预测与当前draft；下一轮DSpark的seed、selected context/count位置依赖acceptance及advance（bootstrap/vllm_dspark_handoff.py:303–390）；下一轮Target输入依赖下一draft。同请求跨cycle不能无条件并行。
- collective结果消费者必须等待本次正确数据到齐；所有rank保持匹配collective顺序。当前rank等待不全是链路传输（Run267）。

**有证据或值得审查的 schedule 限制：**

1. hidden gather→local Q 是人为串行边，Run292已实际放宽。16样本AG中位14.88µs，真实Q核重叠13.75µs，Q链83µs；不能直接乘43层或当E2E收益。
2. main compressor→QLI的先后无直接数学因果，可fork但资源竞争/别名/数值门槛仍未闭合。Run285不构成通过或确凿候选错误。
3. 每rank为非owner请求更新缓存是数据分布选择；Run287仅证明每rank两个owner需完整16输入行，以及某些保守页包络无交。actual native scatter、递归state、跨层typed alias、DSpark/prefix/later-cohort仍unknown。不能删成local12行。
4. 不同请求之间没有自回归因果。当前12请求全层/全cycle齐步、完结slot继续执行，是execution组织。请求微批流水线、inactive算术跳过、设备resident readiness可能合法，但会改变weight reuse、MoE tile效率、CP负载与HCCL顺序。跨请求并行不等于8卡有闲余资源；禁止把两段时长取max作为收益。
5. Target/DSpark间部分旧context投影若不依赖accepted选择，可能预计算/预取；真正依赖seed和选中hidden的计算不能提前。现有源码尚未证明该局部consumer闭包，不宣称整段可并行。DSpark是parallel drafting，不应虚构7次串行draft模型调用。
6. MoE shared/routed分支、dispatch/combine与邻接HC/通信可细分，但 FusedMoE 可能已包含重叠；需对应冻结执行分支再建边，不能把Python顺序当设备串行证据。全cycle Graph/persistent可压缩submission，不能消除前述RAW或必需HBM。

## 4. 已完成的新离线 acceptance/parking 分解

产物：`astra_existing_acceptance_screen.json`。读取 Run287 全部40 ownership JSONL，35个非rank0/cohort进度对照与rank0完全相同。由相邻 cycle 的 emitted_token_count 差值，最后cycle用对应Runtime staged总数补齐；计数均0..8、inactive均0，无违例。它是受probe影响的轨迹，不是Run99正式轨迹或时间基线。

定义 C=执行cycle数，N=12C；A=执行前active slot-cycle总数；S=staged输出（含overshoot）；U=最终有效输出。严格计数恒等式：
8N−U = 8(N−A) + (8A−S) + (S−U)。

Run287五cohort：C=1496，N=17952，A=14963；U=61440，S=61926。
三项分别为 parked capacity 23912、active未兑现capacity 57778、overshoot 486，总计82176。
inactive slot-cycle=2989，占16.65%；各cohort11.39%–20.56%。
S/A=4.13861；U/N=3.42246。许多后期cycle只剩1–6活跃请求，原96行仍执行。
**这些是输出位置/槽位计数，不能称对应FLOP、HBM或时长损失。** active未兑现capacity包含draft不匹配等，不是已经证明可提升的接受率。被parked请求可能仍触发共享weight读取；跳过它们未必缩短最慢rank。

Run99 formal三个pass的有效输出/执行slot-cycle分别3.36565、3.37954、3.39635；overshoot318/356/319。原报告只有分窗均值，没有完整active/count历史，不能将Run287的16.65%移植到Run99。fixed_serving.py:179–187的acceptance_window_means分母包括parked槽，不能作active条件接受率。
Run99 initial_output_counts=0，四cohort各12请求×1024，至少512 target-cycle的宽松计数下界仍有效。1217/1212/1206与512之差不是可消除cycle数，更不证明“只要完美接受就能翻倍TPS”。提高proposal质量、改变verification组织可能保持最终语义，但保持DSpark7合同及必要算术的边界需明确，不擅自假设更换draft模型。

## 5. 下一项最高信息价值实验

**完成当前 H003 同配置A1/B后，优先 one-c4-layer request-owner compact update fixture。**
它直接挑战更大的当前工作集合，不继续扫描小kernel。实验单位是完整layer入口的真实tensor/metadata/state；先A/A建立typed局部可重放，再只改indexer cache更新96→16完整owner行，保留QLI原12请求索引及main attention/collective。

必要前置只采尚缺字段：native CompressorMetadata实际有效slot与typed state读写范围，映射原request→compactrequest；保留所有owner历史及split请求完整8行。fixture应独立私有cache，执行后还原，不发布prefix，也不扩展为持续省写。局部fixture无需先证明永久删除的全部cohort生命周期，但绝不能据局部通过晋升永久省写。

可证伪判据：
- owner key/scale/state、QLI topk、Sparse输出在A/A基线上满足冻结数值门槛；首次分叉定位在更新还是消费者。布局/alias/递归state无法封闭即停止此版本，记录unknown原因。
- 真实16行tiling、必要权重读、HBM/并发成本和 update入口→Sparse完成均测量；row数下降而资源/critical endpoint不改善，就否定“该compact实现有价值”，不否定所有ownership架构。
- 通过后再连续rollback/parking/前缀复用闭包与all8 FULL Graph；进入完整Target/cycle及重复正式E2E，不能从单层counter获益直接KEEP。

若该fixture闭包成本被证明过高，第二选择是**保持slot编号、96输入/CP布局不变，单个token-local MoE分支对inactive请求跳过工作**的固定尾部fixture；先核实router是否跨token有capacity/drop行为、shared/routed汇合与masked输出不被活跃请求消费，不能简单“零输入”等价跳过。此方向可影响GMM/共享专家等更广工作，但负载、tile效率与相同TP图形风险高于当前H003。refill/请求迁移及prefill并行应作为后续独立架构方案，避免同时改变ownership生命周期和算法轨迹。

## 6. 置信度与收敛条件

高：Run292真实局部overlap；Run287查询归属、进度/parking分解；同请求关键RAW；当前正式数值。
中：request-owner/parked维度有比单层H003更大的工作组织空间。
低或unknown：合法可删的完整traffic、真正并发容量、全层扩大收益、改善DSpark条件接受率的上限、跨请求流水线净收益、最终TPS界限。

下一轮模型应同时报告资源库存、合法边集、已实现并发能力、active条件接受率/parking计数与Product边界；没有一项单独足以作“接近极限”的裁判。
