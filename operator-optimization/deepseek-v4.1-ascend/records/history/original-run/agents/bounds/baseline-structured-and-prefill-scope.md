# 服务端结构化输出条件与输入 TPS 口径

## 当前源码能证明什么

- 依据已与历史哈希一致的客户端构造代码和冻结测量配置，正常路径请求不添加 structured-output 等相关字段；适用条件及客户端输出 usage 来源详见 `stage-101-local-benchmark-client-review.md`。这不直接证明服务端请求对象的内部标志。
- `draft-constraints-snapshot/structured_output.py:294-297` 证明：如实际 `request.use_structured_output == False`，`validate_tokens` 原样返回 draft；若为 True 则调用额外逻辑与 grammar，不能由目前快照覆盖。
- `metadata-path-snapshot/vllm_scheduler.py:1541-1542,1855-1872` 以 `request.use_structured_output` 决定将请求归入 grammar mask 集合；`:2404-2459` 更新 draft 时调用 validator。它们没有定义遗漏 HTTP 字段的服务端默认值，也没有定义该标志如何从 API/采样参数构造。
- 因此目前的严格链条是：**payload 无相关字段**（条件性已支持）→ **服务端 sampling / request 如何解释**（UNKNOWN）→ 若 `use_structured_output=False`，validator 不扩张（源码已支持）。不能跳过中间一环。服务级 `StructuredOutputsConfig` 不表明每个请求已开启或关闭。

### 最小剩余证据

只需优先检查两条已冻结源码的明确 import 目标，并保留双容器路径和哈希：`/vllm-workspace/vllm/vllm/v1/request.py`（scheduler 的 `from vllm.v1.request import Request`）与 `/vllm-workspace/vllm/vllm/sampling_params.py`（runner 的 `from vllm.sampling_params import SamplingType`）。分别核对 `use_structured_output` 的构造、依赖，及采样参数 structured 字段默认；再仅按真实引用确定 API completions 请求协议文件，核对其省略字段如何转成 sampling params。仅有其中一段不能证明完整 API→Request 链。当前没有这些双容器快照，不虚拟其内容。

## 本次能否给出 prefill 输入 TPS？

**不能从现有资料严谨给出模型实际 prefill 计算 TPS**。每轮记录 8 个请求、每请求 32768 个逻辑输入 token，prefix-cache query 增量 262144、hit 增量 235904，即 query-minus-hit=26240，命中率约 89.99%。这个未命中计数是缓存查询口径，不能不经事件和算子口径证明就等同实际执行的 prefill token 数；额外重新计算、chunk、缓存边界及运行实现均影响对应关系。

作为**另一个明确的端到端业务指标**，可以在假设客户端 8 个成功请求各自输入数 32768、且这些请求归属于对应 AISBench 窗口时，定义 `已完成请求的逻辑输入 token 总数 / 从最早请求 POST 前到最晚非 `[DONE]` 数据块记录时间的客户端窗口`：

| 轮次 | 逻辑输入 token | 全批客户端时长 | 逻辑输入计数 / 全批时长 |
|---|---:|---:|---:|
| measure1 | 262144 | 15.6824755 s | 约 16716 token/s |
| measure2 | 262144 | 15.8346661 s | 约 16555 token/s |

这是已完成服务工作负载的逻辑输入计数率，分母覆盖 prefill、decode、网络和并发交叠，分子包含缓存命中；**不是**NPU prefill 速度、输入阶段吞吐、算子效率或不可超越的理论上界。此指标与生成 TPS 一样依赖请求及测量范围一致；server metrics 窗口与客户端窗口不完全相同，不能把二者时间直接互换。

AISBench 标为平均 TTFT 的记录约 770/775 ms，但按已核对的客户端正常路径，计时是从 POST 前时间点到首个非空、非注释、非 `[DONE]` SSE 数据块的记录时间；未证明首块一定包含首个生成 token，不能把它直接称为严格的首 token 延迟。它可能包含排队、prefill、采样与返回等部分，并发请求相互重叠；不能用 `8×32768 / 该指标` 当作有效 prefill TPS，更不能得到硬件上界。

要得到实际 prefill TPS，需定义要计数的是逻辑输入、未缓存 token，还是各层实际执行 token；记录每个相关请求及设备的 prefill 开始/结束、具体计算 token 数、共享缓存和并发/分块，以及事件到对应 DP 的归属。以覆盖所有 prefill 事件的规定墙钟区间算聚合吞吐；以各 NPU 或模块活动区间另算设备效率，显式处理跨请求和 decode 的重叠。取数及 profiler 开销需注明。当前 trace/数据未闭合该口径；单次观测效率也不是物理硬上限。
