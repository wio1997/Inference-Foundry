# 整 inactive-DP 与 Gloo 就绪交付判断

本轮只完成这一段的源码、已有事件和条件价格判断。Goal ACTIVE，全链路未扫全；不保证已找到全部融合。每个替换原语逐位一致；没有新 native 准入、接受基线或实测加速。

## 介绍

从 DP engine 的本地无工作分支出发，沿 `execute_dummy_batch → _dummy_run → target40 → draft3 → 最后输出` 追到消费者。先问能否少做整轮、少发合成 sender、减少 host 交付，才判断局部 Engram/HC/metadata 操作。原始 C8、六普通负载与新 framework-profile C8 分别记账；新 profile 不能替代原 227 活跃波价格。

## 到了哪里

- 统一入口重新核对 171 主机成功，原容器镜像仍为 `ce42d65b…`。20 个源文件中16个与本地冻结件逐 hash 相同，4个新取；又取5个依赖。只代表这些运行树文件，不能证明 loaded bytecode、所有源文件或二进制一致。新取文件不自动等于已读完其调用树。
- 源码链包含 engine 队列、worker 两种执行入口、target/draft DP coordinator、padding/mask、Engram prepare/math、MoE 接收真实 peer 专家、DSpark max-length/None-metadata 分支和 workspace 生命周期。具体范围见 [source review](inactive-dp-source-review.json)。冷启动4个 Triton warmup helper 的内部 kernel 仍部分未读；不把 startup-only 工作算作正常请求收益。
- 原 C8 第3波 DP1 本地 HC 是 M3，MoE 接口仍补到 M416，保留远端真实专家服务。新 profile 则逐 rank 对上 CPU 提交家族与 native Model-ID：DP0 各13轮/559 DFC，DP1 各15轮/645 DFC；每轮 target40＋draft3，每个 DFC 两次 HcPre/HcPost。8个 CPU DP 对是 `[tp,tp+8]`，不是一个16-rank all-reduce。
- 实际 Ascend target 调用 `_sync_metadata_across_dp`，与 draft 共用 INT32 `[2,DP]` owner-column SUM 数据包：第0行 token 数，第1行图模式；target 的图模式取最小值。不能用继承 CUDA runner 的另一个 coordinator 布局设计本补丁。
- `execute_model` 入口推进 profiler step，`execute_dummy_batch` 不推进；step 超过12同步 stop/export。因此“12 worker iterations”不能推出只有12个 GPU pass，更不能直接用总轮数减12认定 dummy 数。

## 结果

| 交付范围 | 状态 | 判断及适用边界 |
|---|---|---|
| 全局无真实工作：host BatchFrame＋target40＋draft3 联合快路 | 待确认 | 比单原语更大的合法候选。新 profile 两个 DP 代表的前两轮无 Engram hash，与无请求 dummy 相符；尚缺实际全16 real-work 标志。按首次 frontend submit 夹住、甚至允许免费抹掉下一真实准备到 backend entry 的有利范围，第二轮上界21.561858ms，P90 1.500668%、整批0.123345%；第一轮上界0。rho=.5、每候选轮新增.1ms 时10.580929ms，只是条件价格。与已有 metadata-host 联合范围重叠，不相加。六普通频率/时序未绑定。 |
| 一侧 idle 的专家 service-only、合成 sender 消除、资源重排 | 待确认 | 本地空不等于 EP 空。必须保留 peer 接收/GMM/combine、cross-rank sync 和实际最后 unpermute 读者。删合成 sender 会改变接收行序/M、GMM 几何及量化舍入；须逐原语证明逐位或保留原几何。旧专家计数仅捕获 graph target，不含这一 eager M416 接收分配，不能借该计数给整 target prefill 定价。更大 EP 重写和六普通负载仍待确认。 |
| 原 C8 一侧 idle 的独立本地 prefix/tail 删除，固定 active 到达、EP 服务与资源位置 | 已停止 | rank8 整 target＋draft252.419277ms，其中43 DFC220.401345ms，补集32.017932ms不是可省时间。全部43次 DFC 的 latest idle entry 都早于 latest active entry；最终 draft 结束也早0.957760～1.369634ms。该固定条件下无 latest-entry-frontier 信用，不再为同一 C8 小配方补 native；不排除改变资源/EP服务的范围。 |
| 新 profiled C8 独立 Gloo backend-only，固定 producer 到达 | 已停止 | 26个共同 ordinal、8对共208对事件，完成时刻差中位.00789ms、最大.19105ms。按每阶段 max TP 的“最后 backend 开始→最后完成”全免，有利 P90只5.586240ms/0.388793%；rho=.5＋每阶段.1ms只1.393120ms/0.096959%。停止该录制 C8 独立 TTFT 1%预算；不剪 host/producer 联合交付、未录 tail 或普通负载。 |
| Engram 全 false mask 直接返回 hidden 的通用代数捷径 | 已停止 | 原公式仍做 `hidden.float()+0*value.float()` 再 BF16。IEEE 源公式存在 −0→+0 位模式差：0x8000与0x0000。只是通用等式反例，非新 NPU 失败，也不声称实际 frozen hidden 含 −0；需要保留符号/舍入或证明实际域。该独立旧 C8 idle 小预算本已无关键到达收益，不补无决策价值实验。 |
| 本轮新原语准入 | 已确认可实现：无新增 | 只确认了源码/事件关系及上述有条件判断，不能将 review PASS 当逐位或模型性能通过。 |

### 待确认范围的具体设计和缺口

全局无工作快路应复用现有 DP target 交付，显式增加真实工作、调用类型和序列/代次字段；采用独立字段扩展布局，不能在未证明范围/溢出契约前偷偷复用 token 或图模式位。真实工作标志来自 scheduler/engine 的真实 scheduled tokens 和队列状态，不能从 padding 数、`metadata is None`、局部 input_batch 空或历史 observer epoch 推导。

所有对应 DP 对必须对同一轮得到零真实工作，且 TP 对之间须有受证明的同一输入序列/决定一致性。现有8次 pair reduce 本身不提供16-rank 全局原子结论；缺口是 executor 分发/TP 一致性证明或新增协议及净成本。若任一 rank 真实工作/有待保留的 GPU 副作用，仍走原几何路径并服务 peer。不能跳过 warmup、capture、profile 内存定型、lazy initialization、计数/错误传播或恢复。

`step_with_batch_queue` 返回的本轮 `model_executed=False` 可能仍在排空前轮真实 future；`_process_engine_step` 也会处理 output/finished/远程 KV 等状态。必须单独排空并退役已提交 input/H2D/cache/mask/unpermute leases，才能复用 storage。全局空闲轮可以不提交新的43次 DFC，但不能据此撤销前轮的最后读者或自行改变各 rank 后续 collective 序列。还缺原真实计算几何不变的证明、engine inbox 接收时刻、全部 rank 真实工作标签、真正可暴露部分以及协议/event/backpressure 成本。普通六负载缺同交付匹配时间与密度；留下缺口，继续下一段。

### 新 profile 的约0.7秒长等待不能直接算收益

DP1 独有 ordinal26～29 全在8个 first-token 之后，无录制 DP0 partner。最大715.282070ms；DP0同步 stop/export 的日志在19:36:45～46，DP1在19:36:46～47；最长 TP0 wait 跨越 DP0 录制末尾。录制边界干扰证据强，但秒级日志不能分解其中多少为 stop/export、多少为其他上游工作。不得把原报告 Free/Gloo 交集整个当生产瓶颈、TTFT收益或吞吐收益。

长共同等待也主要可能是提前到达：如 DP1 ordinal5 本地150.00184ms，对应 pair 最晚 backend 进入到最后完成仅.24963ms。上述有利价格仍包含调度等因素，不能称纯网络耗时或预计加速；调用者 metadata/CPU 准备在边界外。

## 证据与下一段

[完整可复算判断](inactive-dp-causal-frontier.json)、[全16 trace 只读提取回执](inactive-dp-framework-edges-receipt.json)、[客户端回执](inactive-dp-framework-clients-receipt.json)、[frontend时钟](inactive-dp-frontend-clock.json)、[当前源身份](inactive-dp-execution-identity.json)、[额外源](inactive-dp-extra-dependencies.json)、`tools/analyze_inactive_dp_edges.py`。压缩原始选中事件留本地，以原件 SHA 固定。frontend Chrome 时钟采用 `baseTimeNanoseconds + ts*1000`；错误的 raw monotonic 关联已废弃并保留探索记录。

本轮 NPU0、模型加载0、服务请求0、新 profile0、远端写入0；只读原容器源和另一个已完成任务的既有 profile。未更改服务/环境，不把已完成 profile 控制器当当前 live 证明。

下一段沿尚未闭合的 prefix/cache/provider/finished/error 退役交付，先核实冻结 KV-transfer/offload 等 guard，再追实际消费者与跨迭代最后读者。已有高价值联合候选保留具体缺口，不用其待确认状态阻塞剩余源码扫描。
