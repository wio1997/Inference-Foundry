# Challenger：Run244 设备证据后的 H2 裁决

**裁决：PARK H2 性能实现，KEEP 一个离线依赖与成本反证问题；不进入 phase-cohort admission patch 或新的性能 Run。** Run244 已把“PP 阶段计算是否明显分离”从 host 推断提升为设备证据；但“独立 cohort 可实际并行”和“拆批净成本有利”尚未成立。不是永久否定 H2，也不是要求再跑一个泛化 profiler。Current=None，真实完整 E2E Gain=unknown，没有新增代码级 PERF_KEEP。

## 亲读范围与事实

亲读 GOAL_REVIEW、Run244 REDUCE-DEVICE-20261006 的 summary.json、stage_bins.json、decisive_events.json、reduce.py/result.json；重读 native scheduler eligibility、AsyncScheduler、native PPHandler、Ascend accepted+draft 广播补丁、worker PP send handle 及 native model runner 的 PP consumer。只写本文件，没有服务器操作或实验。

归约实际选择 Ascend Hardware track 的 X 事件，以 Task Type 区分 compute/communication/copy/AICPU，排除 control wait 后求区间并集。Stage union 是该阶段八个 TP rank 的“任一 rank 活跃”时间，不是全 rank 同时忙、不是算力百分比、更不是关键路径工作时长。独立把全部 5ms bins 积分，精确重现 PP0 compute=583823us/busy=1191950.25us，PP1 compute=630592.25us/busy=648817us。共同有效 SSE 窗口为1193120.25us；两阶段 compute overlap=42633.75us，占窗口3.5733%。这是可靠的归约内一致性检查；本轮亲读的是带原 timeline array index 的 raw 摘录，没有假称重新加载远端全部约80MB/rank原始 trace。

设备片段显示反相区间：窗口偏移10–35ms的5个bins，PP0 compute约95%而PP1 compute/communication/copy/busy全0；偏移40–60ms则PP0 compute=0、communication=100%，PP1 compute约99%。这比“empty scheduler slots”强得多：被分类的设备工作实际分离。仍不能把该窗口所有缺口或通信持续时间当可消除时间。

决定性 raw 反例：decisive_events 中 PP0_TP0 index289154，stream31，connection_id91011，AivKernel/COMMUNICATION持续33564.409us；index381162持续34157.8575us。PP1_TP0所列最长通信仅85.0985us（index340242），但这只是选中rank/窗口的事件类别，不能据此断言长PP0通信必然全是远端等待。PP0 index262900 的stream47 EVENT_WAIT后出现index262920的IndexCopy及index262924的_post_update_kernel，符合PP结果消费形态；没有host API/communicator映射时，不能把它直接命名为特定request/batch的PP broadcast消费。

三次采集/driver失败与本次保留raw的离线有效归约分别记账。CLI等待失败不抹去实际设备证据，离线归约有效也不把原操作改记成成功性能Run。无需再陷入 profiler 修复循环。

## Top Gap 与路线排名

1. **最高价值代码问题：PP 返回广播的等待与独立 cohort 的执行是否能够真正重叠。** 阶段计算分离已测；约33ms PP0通信区间与PP1计算对应，为具体依赖分析入口。可消除比例 unknown。
2. **H2 admission/phase packing 的净成本。** 只有第1项证明合法重叠且成本界有利，才值得一个最小候选；目前不是已确认最大可消除Gap。
3. **H1 ready-useful-commit优先级。** 沿用Run242限定关闭；Run244的设备分离不能重新证明其成立。

本轮不新增通信优化假说或Runtime重构；第1项是H2能否成立的必要条件检查。若它否定纯admission方案，先关闭该方案，再单独决定是否有证据支持下一方向。

## PP0 提前 receive：支持与反例同时存在

支持逻辑可行性的源码：native_pp_utils.PPHandler.receive 在 broadcast_stream 上等待先前 main_stream，然后发起 sampled payload/combined 两次广播；使用 sibling device group。main_stream 对该slot的等待发生在 get_prev_sampled_outputs 消费T+PP的slot时。PP2下，T+1的独立cohort理论上不依赖T的token，因而没有源码中显式“每个T+1都先等T广播完成”的必然边。AsyncScheduler设置每请求current_step+PP，不等于所有请求必须同相。

但以下都未被源码注释或现有raw消除：

- Ascend COMMUNICATION AivKernel 可能占用独立cohort需要的AIV资源、通信队列或隐藏的流依赖。独立stream/group不等于硬件资源独立；upstream关于NCCL的注释不能充当Ascend并发保证。
- native PPHandler仍在每步开始消费旧slot；新的cohort若错误偏移到T+PP而非可合法的T+1，仍先执行main_stream.wait_event，不能只改eligible数字绕开。
- ascend_worker.execute_model先等待此前_pp_send_work。hidden-state P2P发送、广播接收与下一target的真实设备依赖需要一并闭合。
- Ascend补丁在propose之后把accepted+draft统一发布。PP0早启动receive时PP1尚有必要target/MTP工作，长通信很可能包含peer等待；但“很可能”仍是推断。把receive推迟也可能延迟真正消费者，并不是自动正确的优化。

因此：AIV等待**可能**阻挡独立cohort，当前资料既不能排除，也不能证明。PP0 busy≈100%不是“已饱和不可优化”的证据，PP0 compute≈49%也不是“另有51%计算容量”的证据。唯一可接受的下一步是从现存trace的connection_id/host API/communicator/stream/资源信息构造具体依赖，而不是先拆批后解释。

## 拆批成本何时直接否定收益

用同样有效token、同样MTP工作口径定义基线一个4请求轮次耗时T4。若候选每phase为2请求，令S0(2)、S1(2)为各阶段实际不可重叠服务需求（含该stage的target、必要通信及末stage MTP；不能拿stage union直接代入）。两个phase服务同一4请求轮次的理想吞吐下界为：

`T_split >= 2 * max(S0(2), S1(2))`

这是乐观下界；加入广播资源冲突、host调度、bubble及启动排队只能更慢。若该下界已≥T4，则无须性能patch即可否定该吞吐收益。若半批近似保留整批的权重读取/collective固定成本、使该下界接近T4，任何正overhead也足以让收益消失。不能把“新增工作量≥30ms”粗暴作为通用判据，因为新增工作可能部分重叠；比较应在真实瓶颈资源/关键链上。

反过来，下界<T4仅代表未被否定，不代表收益已证实。现在没有S0(2)/S1(2)的可信数据；4请求trace中某个短kernel或4→2尾部不同上下文，不自动构成半批匹配成本。可先查同源现有证据、Graph shape/固定collective数与权重流量界；如不能给出有利且可信的区间，就保持PARK。即使吞吐界有利，admission延后、长prefill与动态到达导致的TTFT/TPOT、公平性也仍需完整合同验证。

## 唯一下一代码问题与最小反证

**问题：保持原PP2 slot、accepted+draft与KV语义时，T+1 cohort 的PP0首个必要target/collective能否在T的PP返回AIV结束前实际执行，并在同工作口径下留出正的净成本余量？**

现在只做现有artifact上的离线分析：以PP0 index289154/connection_id91011为一个锚，连到实际广播或其他通信API及peer，追踪下一target的流/事件/资源前驱；同时查是否已有同一安装/Graph路径的合法独立工作与长通信重叠证据。没有同Run batch ID须明确不能把Run242 step序号嫁接过来。必要的完整timeline若本地缺失，由既有保存artifact提供；缺项不能用猜测补齐，也不新增Run来绕开Goal Review。

最小否定证据任选其一即可STOP纯admission方案：

- 独立cohort首个必要操作仍有不可绕过的前驱指向T返回广播，或相关AIV/collective资源使它必须等广播完成；
- 两phase服务需求的乐观下界已不优于T4，或加入已知不可重叠开销后无正收益空间。

若明确证明没有该串行前驱、资源并发可行且成本区间存在实质余量，才将PARK改成一个最小原生控制候选，先CPU correctness覆盖slot/generation、KV、MTP、abort、deferred grammar、长prefill与公平性，再按新Reset决定是否值得唯一matched性能验证。当前三份归约没有完成这些必要条件，所以不批准性能实现。

Stop List：不按30ms uncovered/33ms通信直接记收益；不按union计算2倍加速；不改kernel、不绕过PP state或缩减MTP；不恢复H1、不扫参数、不新增并行候选、不做大型E2E。今天重新开始会保留有效设备证据，停在这个具体依赖/成本问题，不继续无边界采集，也不凭阶段交替便承诺phase packing。
