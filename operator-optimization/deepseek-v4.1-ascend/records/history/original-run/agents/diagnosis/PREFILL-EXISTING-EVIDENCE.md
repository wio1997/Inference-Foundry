# 已有基线能否定位 prefill？

## 结论

目前**不能从现有基线和已核查的 trace 得到实际 prefill 的开始／结束时间、实际重算 token 数或 prefill TPS**。我们能复算两轮客户端的发送与首个 SSE data 到达时间，以及服务端缓存查询、命中计数；它们是不同观察窗口，不能直接相除。

## 两轮正式基线的客户端时间

正式基线每轮 8 个成功请求、各有 32768 个记录的输入 token。只读取保存的客户端 JSONL 和时间数组数据库，按每个请求从发送到第一个符合客户端规则的 SSE data 行，结果：

| 指标 | measure1 | measure2 |
|---|---:|---:|
| 8 个请求的客户端发送时间跨度 | 38.734 ms | 39.215 ms |
| 从最早发送到最晚首个 data 行 | 1136.305 ms | 1130.712 ms |
| 首个 data 行到达时间跨度 | 835.618 ms | 829.080 ms |
| 单请求发送到首个 data 的平均 | 769.869 ms | 774.643 ms |
| 单请求发送到首个 data 的最小–最大 | 300.687–1097.571 ms | 301.632–1091.497 ms |
| 整批最早发送到最晚记录结束 | 15.682476 s | 15.834666 s |

每轮 8 个“发送→首 data”区间均互相连通，因此它们的时间并集恰好为该轮从最早发送到最晚首 data 的时长。**这不表示八个请求的 prefill 共运行了 1.13 秒**：其中可能有客户端和服务端排队、网络与序列化、prefill、解码、采样及返回，阶段也可能相互交错。首 data 到达差异是客户端观察，不能单独归因于某个 DP、算子或调度。

## AISBench 的 TTFT 具体记录了什么

与历史哈希一致的 `vllm_prefix_cache_api.py:150,161-174` 在发送 POST 前记录时间，并对每个非空、非注释、非 `[DONE]` 的 SSE data 行，在 JSON 解析前记录时间。`default_perf.py:167-177` 对流式请求将第二个时间点减第一个作为 `ttft`。因此上表“发送→首 data”复算出的平均值与基线报告的约 769.9/774.6 ms 一致，但**记录时间点本身没有保存该行是否含生成文本或 token**；不能仅凭这份数组严格确认它是真正首 token 的到达时间。AISBench 给出的 `TTFT` 名称可保留，但须说明这个实际观测口径。

`base_perf_metric_calculator.py:364-375` 还把输入 token 数除以该客户端 TTFT 命名为 Prefill Token Throughput；这只是按客户端指标定义出的逻辑比率，包含缓存命中与非 prefill 部分，不能解释为设备实际处理 prefill 的 token/s。这里不把这个比值报告为算子或硬件速度。

## 逻辑输入、命中和实际重算

每轮客户端记录逻辑输入总数 262144；各轮单独的服务端计数器快照差显示 prefix cache queries=262144、hits=235904、两数差 26240、命中比例约 89.99%。它们支持冻结基线的负载描述；**26240 只是查询计数减命中计数**，尚无同一请求、每次调度的 token 数与缓存块复用的绑定，不能当成真实执行了 26240 个 prefill token。额外重算、块边界、分片、chunk 和投机等都可能影响实际执行量。metrics 前后快照也不是客户端 send→first-data 同一时间边界，不能用这两个计数去除以上客户端窗口。

单次 profiler 负载的 `evidence/profile-workload-audit.json` 对保存的 8 个请求做了输入一致性核对，计数器与这类负载相符；但它**不是正式无 profiler 基线同一次请求**。已查 EP8 数据库 schema 和字符串名称的范围见 `request-dynamic-metadata-coverage.md`：没有建立逐事件请求、实际长度及阶段的连接。它并不证明其它未查导出绝无这些信息，也不能把已归属 target decode 窗口当作 prefill 工作。

## 最小补证，才能回答实际问题

在同一批请求和已定义的计时口径下，至少需要：

1. 每请求匿名可关联标识、服务端接收与排队/首次调度时间，以及前缀命中、未命中、实际可复用范围；避免读取用户内容。
2. 每个 prefill chunk 的执行起止或明确 kernel/graph 归属、token 数与所属设备/DP，以及是否重复执行或与 decode 并行；保证没有漏数或重复数。
3. 客户端首个**实际含输出 token**的事件、服务端首 token 产生/发送事件，及它们同请求的关联；这才能分开排队、设备工作和返回。
4. 若跨 host/device 比较时间戳，还需已验证的时钟域转换和误差；跨设备任务先各自求并集，不能把并行时间直接累加。
5. 事先约定分子是逻辑输入、未命中部分还是实际执行工作量，分母是请求等待、prefill 时间并集还是整轮墙钟；区分 profiler 扰动与正式无 profiler 基线。资源硬上界及必须工作量另需证明，测得 prefill TPS 也不是理论上界。

## 证据及复算

`analyze_prefill_client_bounds.py` 使用 SQLite `mode=ro` 和 NumPy `allow_pickle=False`，只解析时间数组并输出聚合，不写 prompt、回答、请求 ID 或绝对时间。`prefill-client-evidence.json` 的 SHA-256 为 `38cbba1dde4eb4503617e352372efd35bbbed17033cce361ae720b6d93f7a94a`，记录了源 JSONL、数据库、客户端源码及脚本哈希；与基线报告的整批时长、平均 AISBench TTFT 核对一致。ZCode stage `operator-ceiling169-v2-108-prefill-client` exit code 0，model_verified=true，模型 `deepseek-flash`。过程未启动设备、重跑负载或修改服务。
