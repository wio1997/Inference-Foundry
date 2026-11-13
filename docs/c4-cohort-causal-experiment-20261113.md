# C4请求组准入的单变量归因：观测源码已准备，尚未上机

本文件是[机制审计报告](performance-mechanism-audit-20261113.md)的后续实验设计，不包含新性能结果。固定模型源码仍为`76bb59edf66caef19e81a13bf91b698aba09659b`，不引入未测的完整尾段分批141。

## 1. 只回答一个因果问题

已有三臂数据表明，C4关闭状态复用后仍领先参考23.73%；整次运行中参考与原生普通MTP/图回放计数不同。目前缺的是：请求组准入是否通过减少prefill/decode混批、保持更多uniform decode，产生这部分C4优势。

本次只比较同一个原生实现的cohort开启/关闭。两臂均关闭混合状态保存/恢复，保留3GiB普通池+1GiB状态保留区、raw-logprob取行和所有共同优化。唯一有意差异为`--native-admission-cohort`是否存在。

先只测C4：它是现有归因缺口。每臂保留原有4条工程请求、2条预热、32条1024输入/256输出的正式请求，共38条；在C4完成后退出，不重复已有C8状态复用裁决。两臂都采用相同改变，C4之前的请求配方与原实验对应。

## 2. 必须保持的条件

- 固定SDK、模型、权重、C++扩展及共有helper身份；模型源码与139一致。
- 同一8die卡组、TP8/EP8/DP1/PP1；context2048、最多8请求、每步1792 token。
- MTP5、FULL_DECODE_ONLY及有效24/48图、prefix、async、DSA CP/SP、EP/AIV、共享专家重叠、static/super与prefill MC2。
- temperature0、top_p1、seed0、ignore_eos；相同AISBench C4配置和冻结32条输入。
- 两臂使用字节相同的状态关闭包装及观测helper。不能只看argv相同而忽略入口代码。
- 不以旧vLLM参考的绝对秒数代替这两个新臂的同源码比较；新结果首先回答cohort的干预效果。

## 3. 新增哪些观测

`execution_route_observer.py`在原生SDKRuntime安装后包装已经存在的target `_model_forward`及SDK `ACLGraphWrapper.__call__`，透传同一参数、返回值和异常。它记录：

1. 真实调度跨度、prefill/decode/spec请求数、uniform标记、实际target图模式。
2. 图wrapper确有可回放entry时的target/draft实际回放次数；不会将“配置FULL”当作“发生回放”。
3. 同时记录该调用实际请求ID的SHA256，与服务完成记录关联；请求正文和原始ID不进入观测日志。每rank最多2048条target和2048条图调用主机时间记录。满额只丢观测，不改变推理，但该轮不得宣称完整时间线。
4. 原有`/health`响应中的累计快照；正式阶段不增加健康查询，由原有阶段边界查询保存。
5. 结束后写出`execution-route-observer.json`。保存发生于drain，不逐步写文件。

没有NPU张量读取、设备计时事件、设备同步、额外模型调用或新提示词。host起止时间包含提交/等待，不是设备kernel耗时。记录使用目标机单调时钟；后续与同机AISBench数据库中的正式请求时间窗口及请求身份共同对齐。仅时间重叠不足以区分异步预热，不能据此直接标记正式工作。

观测器先恢复共享MTP observer的graph hook，再进入原来的drain/close，避免破坏已有严格的hook归属检查。代码没有实现新调度器或模型算法。

## 4. 已验证与未验证

已通过10项主机检查：5项观测器契约覆盖原参数/结果透传、target/draft区分与hook恢复、非主机标量拒绝转换、混批host跨度、原异常与事件有界；5项解析器契约覆盖时间窗口分区、时钟不一致、预热/边界排除、不完整日志拒绝，以及落入正式时间窗口的延迟预热和跨阶段混批排除。另完成Python AST检查，核对两臂命令列表仅cohort参数不同。新解析器的时间戳解码与192条已发布原始blob一致。

这些检查没有执行模型数学、NPU代码或实际SDK类。真实SDK字段、所有rank关闭、实际图模式与观测开销都未硬件验收。当前0新增模型加载、0部署，NPU阶段预算尚未分配。

运行前需把同一[入口包装](../validation/performance-attribution-20261113/entry_state_off_route_observed.py)放入执行包的`overlay/postscore_profile_entry.py`，将[观测helper](../validation/performance-attribution-20261113/execution_route_observer.py)放在同一overlay目录；旁边的`DeepInfer/`必须是固定源码。文件不是可以脱离执行包直接启动的服务脚本。

## 5. 运行有效性与采用/否决

### 时钟记录与正式窗口解析

新增只读主机时钟身份：boot-id哈希、time-namespace偏移哈希、monotonic/perf_counter的实际实现。客户端与模型进程必须对应同一单调时钟；不能用本机控制台的UTC时间替代目标机事件时间。

在负载生成容器内、正式AISBench之前，用标准库模式记录客户端时钟；这不会导入torch或初始化NPU：

```bash
python3 -B -S /task/<stage>/overlay/execution_route_observer.py --clock-output /task/<stage>/native-run/aisbench-c4-client-clock.json
```

应在正式请求前将其与原有health响应里的`execution_route.clock`核对。38请求的C4-only服务会在最后请求结束后关闭，因此不依赖结束后的HTTP查询；等待原有drain写出各rank的完整关闭日志，不增加保活请求。

收集后运行：

```bash
python3 analyze_cohort_c4.py --run-root <collected-run> --client-clock <client-clock.json> --output <analysis.json>
```

解析器只读原SQLite/NumPy时间戳，核对32条正式请求、官方汇总、38条总请求、8rank退出释放、实际cohort状态、状态复用关闭及缓存预算。它按正式客户端第一开始/最后结束时间分为窗口内、窗口外、跨起点、跨终点和跨两端，保留边界调用。再将每个请求的首次target调用时间与38条服务完成记录关联：必须恰好32个请求首次进入target时处于正式客户端窗口，且满足1024/256长度；否则拒绝分析。不依赖请求完成顺序。窗口内仅含正式请求的调用才进入正式计数，预热或跨阶段混批单列；即使预热后续调用延迟到正式窗口，也不会混入正式计数。host时长始终不是设备耗时。

客户端/服务器时钟来源不同、helper身份不同、日志缺失/丢弃时拒绝形成该归因分析。实际SDK日志尚未生成，当前通过的是解析契约和历史时间戳兼容性，不是新NPU实验。单臂输出状态明确为`one_C4_arm_analyzed_not_causal_verdict`，必须与另一臂结合后才能讨论因果贡献。

先完成一臂工程门槛，再运行32条正式请求。任何一项不满足都保留失败记录，不能补写通过：

- 实际模型/扩展/依赖、rank映射、共有配置符合固定身份；无计分期迟编译。
- 状态保存/复用/复制均为0，缓存总预算和分配相同。
- cohort开启有实际准入组、全部收尾；关闭时服务回执明确为continuous batching，cohort统计未伪装开启。
- 观测器metadata错误、模型异常及事件丢弃均为0；所有rank关闭且原有hook归属校验通过。
- 32/32正式成功、32768输入/8192输出；38条总请求完成，8rank退出并释放资源。

随后逐档窗口计算：target FULL/NONE、纯prefill/混批/decode、target/draft实际回放、普通MTP调用，以及端到端秒数、TTFT/TPOT和完整输出一致性。

裁决不预设成功：

- 关闭cohort恢复更多混批/非FULL，并显著增加C4端到端时间：支持此机制的因果贡献，使用新两臂的实际差值量化，不能直接搬用原24.13%。
- 执行路径改变但耗时没有对应改善：说明它未形成足够的关键路径收益，否决“cohort解释主要优势”的主张。
- 路径和耗时都基本不变：关闭这一解释；不靠继续调参数凑答案。
- 差异接近未标定波动或工程失败：结论为尚未证实；不把单次结果包装成稳定根因。

本设计最多两个有条件的模型加载，不增加优化变体。资源条件与两臂收尾窗口确认后，才给本归因阶段分配有界执行预算。任何后续测量必须回应这项裁决留下的明确问题。

## 6. 当前外部前提

已只读查明：所需8die全部在现有CANN池策略中有其它任务租约，没有DeepInfer租约；三次独立只读复核均未取得该许可。现场入口的`permission()`要求中央池签发的有效key/lease/device身份，主机策略有效期上限60秒，需由中央池持续续发，不能靠改本地登记或一瞬间没有NPU进程取得长期窗口。

已向用户请求该池的预约命令/API或管理会话位置。在取得协调入口前，不改池策略、转移其它租约或抢占其设备锁。此处是实际资源协议缺口，不是模型正确性失败，也不是本地主机契约检查证明了可上机。

## 7. 材料

- [准备清单、SHA256与两臂计划参数](../validation/performance-attribution-20261113/cohort-observer-preparation.json)
- [观测helper](../validation/performance-attribution-20261113/execution_route_observer.py)
- [同一状态关闭入口](../validation/performance-attribution-20261113/entry_state_off_route_observed.py)
- [5项主机契约检查](../validation/performance-attribution-20261113/test_observer_host_contract.py)
- [正式窗口解析器](../validation/performance-attribution-20261113/analyze_cohort_c4.py)
- [5项解析器契约检查](../validation/performance-attribution-20261113/test_phase_parser_host_contract.py)

上述清单中的命令是计划值，尚未执行；实际启动后还必须保存真实argv、有效配置和运行回执。
