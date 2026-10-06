# Performance Research Reset — H6 bounded native validation

身份：branch glm5-3-autonomous-20261001 / HEAD f22efaab45eeeb3b967931124a6f8f19f9736e85 / GLM-RESEARCH-RULES-v2 / rule commit93133edc6567a2ecaac7754b3c9aab1537ca7fe3 / active point GLM-OPT-0002。新脚本及证据尚未提交，实际被测文件SHA由frozen controller spec与native artifact固定。

产品与Current：GLM-5.3 W8A8标准PD、真实动态请求和完整外部功能，权重/data/tiankuan/wio/GLM-5.3-w8a8。Current=None，active stack空，H4/H5正式INCONCLUSIVE/PARKED。参考CURRENT_PERFORMANCE_STACK、Run249及CHECKPOINT164/165；此小规模验证没有标准动态workload/SLA/有效容量发布价值，不能提升Current。

最大可信代码Gap：模型内部eager host supply，MoE为最大定位region。Run249 exact1901 HCOM pairs中的1896/1901 latest host=latest device，latest-launch→device median24.825us；主要长等待由晚host提交/peer dependency放大。ON的pre-enqueue747ms不是OFF265ms/token的可加预算。精确Scheduler/Core/Executor/ModelRunner entry/exit仍unknown。

唯一活动假说H6：两个MC2 V4 capability predicates在每层重新查固定库内API/WorkspaceSize地址；非模型必要算术、路由、通信、executor或workspace。Production mc2_capability.patch只缓存两个static const bool。Run254 raw SHA63be953de17ebefce743738bae3b6d7780a84373d52a147d936f480c8e70d803中的248 CPU samples全部归到actual binary的两MC2 predicate callers，dispatch134/combine114；binary BL/literal证据和源码pin均核对。Native CPU helper并发positive/negative检查已过，51us/次量级和38.593ms机械参考不是模型预算/收益。

隔离native：same torch_npu5dd8ef3f9b375b5ae4a83538d5785754148c3302/op-plugin8b9c8534fa41eff367a41c155843daa530ab3a08，MC2-NATIVE-BUILD-CPU-20261007-E已compile/link inner0，candidate SHA0dc998d25e765af618090c4d69b395f05c1c072421917fa04c35f78deb7f9258，104480408 bytes，未安装、未初始化NPU。两predicate使用common-binary stock0/cached1控制，所有分配、动态路由、CANN/计算/通信参数不变。Cached冷初始化和mode读取是两组共同成本，warm请求排除。

CPU import A暴露original _C RPATH优先于LD_LIBRARY_PATH，明确failed gate且没有发模型请求。B将字节不变的_C副本和原Python源symlink放独立mirror；原RPATH以mirror的lib目录定位candidate，其他安装文件不改、无LD_PRELOAD或指令patch。原_C SHAe04d0028f79428e9c27fee1e6ab88e70abcbf319354ef821719575e936bf859c。模型Run必须等待B新进程只map候选、两schema与四V4符号相同、NPU_initialized=False。实际audit identity写入Run255 functional_plan和controller spec。

Hypothesis：消除两处重复能力查询能稳定缩短D critical path；同一candidate binary/same resident D16 workers的mode0/1可隔离该机制。

Distinguishing evidence：先mode0/1各一条2334-prompt/8-token golden及一条58-prompt/23-token自然EOS golden；exact IDs/chunks/content/stop、完整external KV hits必须相同。逐rank核验库hash/实际map、two static bool guards+values及实际mode byte；不读任意模型内存。随后A1/B1/A2/B2每phase warm8排除、两短8和一完整23。P249固定，TP16/EP16/DCP16/MTPK1、eager/profileOFF、资源和API合同不变，无参数扫描/large profile。

Decision table（Run前冻结）：

- CPU loading/schema/all-rank witness失败：INVALID attribution gate；不进入或立即退出timing，保存raw并恢复original D，不将其记为cache性能回归。
- golden IDs/chunks/content/KV/stop或模型运行失败：REJECT candidate model correctness；终止比较，恢复原库D并核验all16。
- 两次paired complete D wall与TPOT均降低，且D wall saving两次都大于max(|A1−A2|,|B1−B2|)的complete D wall baseline/patch drift：限域positive mechanism证据；仍需标准动态SLA和完整功能合同后才可PERF_KEEP/Current。本Run不自动安排大型验收。
- complete无一致收益、反向，或saving落于上述drift内：INCONCLUSIVE/PARKED或明确重复回归时REJECT；不扩张参数/请求扫描，继续原raw/source的更大eager producer代码问题。
- 短8、自然EOS23结果或P贡献不同：分项报告，禁止把P波动计作D代码收益，也禁止外推全局最大removable预算。

执行保护：frozen unique controller/双锁，仅退休fresh确认的owned D253；candidate library隔离加载，原P/安装Python/native不改，guard核验既有67 source hashes。finally单独original-native recovery epoch、warm、all D16 original maps/hash、P original owner和source/idle核验。恢复失败保留failed/remaining-owner证据，停止不明owner操作。

Astra：YES，长期无formal codeKEEP且最大eager Gap继续下钻触发；已有Challenger亲读source/raw核查引用CHECKPOINT164/165、EAGER_REPLAY_SOURCE及GRAPH_ALIGNMENT_SOURCE。新graph sizing/key/dummy阻塞的独立复核已接受，但不激活第二候选或改变graphflag。先完成H6 correctness→matched A/B→自然EOS E2E，再回到logical/physical padding消费者。
