# DSpark 三层 body 与 TP/EP 交付判断

公开[完整价格摘要](dspark-three-body-price-summary.json)保留227波价格、每请求净情景、身份与限制；3696行完整关联在摘要所列本地文件及SHA，未冒充已公开的原始记录。

介绍：冻结版本、CANN9.2.0、910C、原C8与六普通负载；**每个替换原语逐位一致**。本轮独立从三层真实调用到消费者，再复用匹配后的历史记录。全链路 Goal ACTIVE，本页只关闭一段源码/价格判断，不能保证已找出全部融合机会。171 本次 `--check` 认证失败；以下为本地固定源码和旧 profile 分析，没有新的运行身份核验、设备验收或部署。

到了哪里：`dspark.py:249–276` 的 own embedding→HC4/SP→三次 decoder→collapse/SP AG，接 `llm_base_proposer.py:1487–1576` 的采样选行与 base head。每层真实路径是：

```text
HcPre → BF16 RMS → TP8 SP AG
 → 本卡 CP Q + 全局 query KV/SWA writer → noncausal SWA/SMLA
 → 逆RoPE → TP head restore/alltoall → OA/OB → SP RS → HcPost
 → HcPre → RmsNormCast（BF16专家输入 + 原FP32 gate输入）
 → gate/top3 → EP16 DFC（8本地专家） ∥ 已有 shared auxiliary-stream 流水
 → 原BF16 shared+routed add → HcPost → 下一层
```

三层各有权重、HC系数、query/cache和状态；没有 target 的long/index源或Engram。草稿128/top3/本地8，不继承target384/top6/本地24或M416的准入。实际MoERunner注册替换为AscendMoERunner；返回tensor已由`_foreach_add`合并，模型里的tuple fallback不是本路径的额外一次all-reduce。SP reduction hooks也不能重复计价。共享输入准备已在router前提交，Gate-Up已在router事件后、routed TopK/dispatch前排队；“再增加同一重叠”不是新收益。

结果：以下状态均限定具体配方。新候选**没有一项达到“已确认可实现”设备准入**。

|范围/配方|状态|判断和具体缺口|
|---|---|---|
|三层同原语提交整合/body graph + shared/routed资源安排|待确认，值得深入|整段整批预算仍有量级；保留全部collective顺序/数值几何。已下钻到capture切口、persistent metadata、实际wrapper同步与DFC设备计数，见下文。缺原M1/2/3各层权重逐位、实际capture/replay、共驻争用和净成本；六普通whole-body匹配时间未齐。|
|跨shared/routed共享输入量化，纳入上述联合方案|待确认|原host tiler在M1/2/3为key21000、3/6/9 AIV，每AIV一个top3路由；原行的FP32 abs/max/div→FP16截断→INT8舍入被3个AIV重复执行。共享流另做`npu_dynamic_quant`。DFC API目前只收BF16 x，不能直接喂已有INT8。必须比较INT8值与FP32 scale所有位、零/特殊值/invalid行并保留排序；不能借target位验收放行。重复次数不是耗时，不据此启动孤立native。|
|query/sender为零就跳过本卡DFC|已停止|本卡仍接收其它卡专家任务，且每层EP协议必须参加。原C8活跃波中三层分别327/483/437个零sender窗有recv。|
|空expert等于尚未删除的GMM计算|已停止该收益口径|源码`currentM>0`已守卫两段GMM，`dequantSum>0`也已守卫激活；不能把空expert权重矩阵当新增免费计算。原空credit协议变体native价值失败，不重跑原配方。更广协议重写并未被此停止项穷尽。|
|把三层当相同/线性层直接合并或缓存全结果|已停止|各层权重、HC/归一化/激活、RoPE、context均不同；第一层重复noise输入不等于整层输出同值。常量前段缓存仅可另限到pre-RoPE，仍需原M/tile/行位置逐位，不能改变GEMM M后声称等价。|
|只消除非DFC typed小算子、独立冲原C8首token1%|已停止同一小预算|匹配四个eager carrier的旧整段“其它typed全免费”仍仅0.4467%；更大联合通信/提交/吞吐范围不受此小项剪枝。1%沿用排序参考，不是用户硬门槛。|
|把typed/HCCL并集外时间全部当CPU可省，或据此否定graph|已停止该定价口径|并集外可能有DMA、runtime、同步；DFC typed时间也含EP等待，提交重排可能改变内部等待。需要真正到达/消费边，不能靠complement判定。|

## 原C8完整parent价格

`dspark-three-body-frontier.json`由脚本可复算。3696个rank窗逐一关联原first-HcPre、三F21/last-HcPost、base-head首GEMM；全部旧DB SHA与原crossover一致。227个active波保留16个EP参与者，共3632窗，包含8个idle DP peer窗；3624才是rank自身active，两个数不可混用。11088次草稿F21均与旧计数关联，active波内为10896次。

价格起点是first HcPre，终点是base词表GEMM开始，包含三body与collapse/归一化/选行等尾段；不包含更早input/context、base词表GEMM和五步Markov。保留原观察器与原调度，按保存的16rank时钟约束计算latest frontier，并有利地传播截止客户端deadline之前的所有波。必要计算、通信、暴露和替换成本在“全免费”中被放松，**不是硬件通用上界或预计可实现收益**，也不能与host/context/score预算相加。

|条件|原C8 P90首token净敏感性|原C8整批净敏感性|
|---|---:|---:|
|整个parent免费、零新增成本|21.252771ms / 1.424587%|663.541961ms / 3.569201%|
|可移除且暴露比例25%，新增0.1ms/波|4.813193ms / 0.322631%|143.185490ms / 0.770197%|
|比例50%，新增0.1ms/波|10.126386ms / 0.678778%|309.070981ms / 1.662497%|
|比例75%，新增0.1ms/波|15.439578ms / 1.034925%|474.956471ms / 2.554797%|

新增0.1ms/波时，达到历史1%排序参考至少需要原parent约72.55%的可移除暴露比例（P90）或31.44%（整批）。这指导联合范围优先面向整批收益，**不能把50%当测量值**。原四carrier“保留原F21 service、移动其入口”的敏感性另为P90约1.01265%，不等于整个EP计算可删除。六普通缺whole-body原时间，不能把另一112窗口residual cohort或v164 TP等待搬来当匹配数据。

## 专家数据与关联条件

三层每卡接收行数中位5/5/2，最大36/41/39；八个本地expert中零expert中位6/7/7。计数显示小M、偏斜和控制开销值得与提交/资源安排一起判断，不能从接收行数直接估elapsed。每个active native波按原有序关联时，三层各有total_recv = 3 × total_sender。

**原observer epoch不是全局序号。** 新增“16rank同epoch”断言失败，已保留error Run；681个wave/layer组均有两个值，DP8各自一致，例如首真实波DP0为15/16/17、DP1为53/54/55。修正后依据实际HcPre TASK_ROW、序列、snapshot和head边界关联；96个三层窗缺独立相邻target anchor的原限制继续保留。总量守恒与同bank也不升级成物理跨卡协议证明。

## 值得深入方案的实际实现边界

1. 提交整合先切在`_run_merged_draft`的`build_model_inputs_first_pass`之后、`self.model(**model_kwargs)`，保留外部context生产和原完整M、三次body原语；输出collapse/SP gather仍由原model执行。parent价格还包括model返回后的sample/index/norm，不能自动全记给这一切口。随后若扩大到head-input，只能连同原选行和norm再评估，五步反馈仍在外。
2. `DSparkProposer:73`强制eager，base的ACLGraphWrapper包装条件因而不成立；仅翻flag不是可用实现。`build_for_drafting`仍走原build。现有V41 builder已具UNIFORM_BATCH、固定SMLA buffer，noncausal indices/length也已有persistent副本；这些是可复用机制，不是“DSpark已支持capture”的证明。CP0-query分支、num_reqs/QSL形状、global-cache与local-query视图、每组query slot、mask和dummy模式仍决定capture家族。
3. 不能只按SP M=3选graph：C8 active波M1/2/3分别32/144/3456窗；相同M3仍可有不同own B、DP最大B、CP实际Q和metadata缺省分支。正常own B/DPmaxB、真实空query、warmup/profile/dummy、设备metadata等待、error/teardown必须分别从实际caller派生。Bmax1..4的14个组合只是设计枚举，不是当前准入家族。
4. ACLGraphWrapper只对**位置参数Tensor**记录地址；现调用用kwargs，不能靠空address列表证明稳定。要显式约束IDs/positions、三组cache/slot/QSL/seq-lens/indices、输出与workspace的storage/offset/代次。FULL replay已有条件`current_stream.synchronize()`，且shared辅助流须正确加入capture和完成等待，新增同步成本属于h，不能假定为零。
5. DFC适配层仍走原`EXEC_NPU_CMD`。设备`HcclShmem::CrossRankSync:166–182`在执行时load计数再+1，并非Python捕获一个常量epoch；同实现原target已有captured replay关联。但这不证明草稿EP16/三bank/辅助流capture已通过。末尾CrossRankSync在unpermute之前，offsetD直到unpermute末读才可复用；graph/megakernel不能把这个屏障当arena退役。
6. 真正megakernel仍需实现TP exchange/RS、EP发布/跨core屏障与SMLA依赖，不能在一个kernel内“调用现有aclnn算子”就获得融合。shared共驻可复用4AIC/8AIV方案思想，但M416真实权重准入不覆盖M1/2/3与三组draft权重。优先保持每次INT32/FP16/BF16舍入、top3原序FP32和BF16 merge/HC边界，再谈消除物化。

下一条：继续真实V1 worker的warmup/profile/capture/dummy和异常退役caller，核目标与草稿如何进入同一调度波、metadata与cache初始化/复用，补上本页graph家族入口；随后回到尚未独立闭合的prefix/provider传输退役与剩余合法分支。此页具体native/净成本缺口留在待确认，不为填表重复小项测试。

证据：`dspark-three-body-frontier.json`（完整3696窗、价格与计数）、`dspark-three-body-source-review.json`（源范围/资格）、`dspark-routing-host-tiling.json`（离线未修改host代码结果）、`dspark-body-exploration-notes.json`（失败与更正）。新NPU0、模型加载0、服务请求0、新profile0、远端写入0；无环境/服务变更。
