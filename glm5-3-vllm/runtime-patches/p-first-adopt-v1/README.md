当前query-only原型不采用：干净同实例C1 ABBA中TTFT倍率1.01566、最后token1.00282，未观察到净收益。2输出dispatch2→1，但原冻结64输出仍13/14次不减。24题四路数值均24/24，PD control/adopt完整64序列仅14/24一致，严格通用逐token等价未通过。负结果与限制保留，继续研究P已有未验证草稿的交接。

该实验将D首次有效query从最后prompt token推进到P已知首输出y1。原生尾行探针已在N256/257、两端各16rank匹配7488条记录，仍未独立证明完整前缀。

原生KV接收完成后，scheduler把y1真实append到Request history，computed保持N；NewRequestData显式携带初始output IDs，worker构造真实CachedRequestState时复制该history，下一query为y1@N。普通Async只处理新D samples，避免把外部token当placeholder完成。y1仅prepend到首个正常D输出的wire，不提前发布、不核对/删除一个“重复y1”，完整输出预算含已采纳的y1。

全局`GLM53_P_FIRST_ADOPT=1`且单请求`kv_transfer_params._glm53_adopt_first=true`同时成立才进入。当前支持N256/257、max_tokens>1、greedy/min1/ignore_eos、文本cold完整remote receive；stop、logprobs、penalties等不支持条件回原生。max1明确回原生，不在KV晋升内结束请求。原native history/stop/free/placeholder逻辑继续处理D新samples，取消清理pending wire状态。

构建器核对4份实际镜像原文件SHA，输出scheduler/DTO/GPU worker及仅附加host trace的NPU worker。每请求开关让control/adopt共享同一个P/D实例；相同配置、输入、预算和采样下比较完整token序列与实际target `_model_forward` dispatch次数，不能把首批6行或draft计数当整请求前向次数。trace只读既有host input-batch metadata，不新增tensor复制、kernel或等待；它不是设备input readback、kernel数或设备时长。创建D run下`adopt-trace/disabled`可停止trace文件写入，再测有限端到端性能。

实际无NPU镜像CPU检查已完整导入Request/SamplingParams/CachedRequestState，并执行真实MessageQueue.enqueue/recv方法（socket recorder），验证初始history与独立copy；没有真实IPC、完整Scheduler/KV manager/InputBatch/GPU回放。失败检查原件保留，原先直接typed MsgpackDecoder的假设已纠正，实际MessageQueue源码另有SHA收据。

`run_pairs.py`先以原生控制建立同实例参考，再比较候选完整token IDs、usage及同一客户端单调时钟。`prepare_cases.py`冻结24项数值/提取任务到256/257长度，保留独立numeric scores；它们不代表完整模型精度。开启复杂采样、C4/热缓存、生产SLO或默认采用均需后续证据，不能由少量匹配推断。

当前构建/CPU/启动与请求证据在[本轮目录](../../evidence/20261018-p-first-adopt-device-v1/)，资源/CID和持续执行授权以[HANDOFF](../../HANDOFF.md)为准。P保持加载，只替换D以安装这4份overlay；不存在每阶段重新授权或固定25分钟停工要求。保持全SDMA原生Mooncake直传配置、原模型/精度和既有算子。
