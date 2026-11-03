# 两轮无 profiler 基线：客户端逐请求时序复算

来源为 `benchmark-20261028/raw-results/results/measure{1,2}` 每轮唯一 `*details.jsonl` 和 `db_data/*.db`；每个 details 行通过 `time_points.__db_ref__` 唯一关联 SQLite `numpy_store.id`，BLOB 用 NumPy `allow_pickle=False` 读取。每轮8个不同 data_id、8个不同引用；16/16 成功、输出 1024、时间点有限且非递减。`recompute_baseline_latency_anonymized.py` 保留源码散列与匿名结果 `baseline-latency-anonymized.json`，不输出内容或原始 id。两轮各按 data_id 排序后赋匿名位置 1–8，两个 id 集合相同；**相同位置仅表示同名数据位置，不证明输入相同或关联同一服务端请求**。

计时口径来自与历史哈希匹配的客户端 `vllm_prefix_cache_api.py:140-185`、`default_perf.py:167-183`：start 是 POST 之前，first 和 last 是收到的非空、非注释、非 `[DONE]` SSE 行解析前的时间；本文把 `first−start` 称为首条已记录 SSE 延迟，把 `(last−first)/(1024−1)` 称为客户端每输出 token 的平均时间代理；二者不等于已验证的服务端 TTFT 或逐 token TPOT。单请求 210–227 条 SSE 记录，而输出均为 1024 token，不能把每条 SSE 当成一个 token 或引擎步；首条是否携带 token 在该计时证据中未验证。last 也不是整个 HTTP 完结时刻。

| 观察，单位 ms | measure1 | measure2 |
|---|---:|---:|
| 8 个 start 的最大跨度 | 38.734 | 39.215 |
| 8 个 first 的最大跨度 | 835.618 | 829.080 |
| 首条 SSE 延迟最小–最大 | 300.687–1097.571 | 301.632–1091.497 |
| 客户端平均时间代理最小–最大 | 14.011–14.691 | 13.737–14.831 |
| 8 个 last 的最大跨度 | 532.959 | 1206.661 |

两轮按 data_id 排序的启动顺序均为 1,2,3,4,5,6,7,8；两轮 first 到达顺序都为 1,3,2,5,7,4,6,8。最后 SSE 顺序分别为 1,3,5,7,4,8,2,6 和 2,4,7,1,3,6,8,5。相同位置的首条 SSE 延迟差异范围 -6.074 至 +19.844 ms（仅两轮），但两轮各自首条 SSE 延迟取值跨度 796.884 和 789.865 ms；计时分布中的这种复现性提示优先检查首响应阶段的按请求/路由分组行为，不足以断言排队是根因，也不能断言完成时间差是计算差。

八请求详情既无 `cached_tokens` 字段，`extra_perf_data` / `extra_details_data` 均为空；服务 metrics 仅提供每轮聚合 cache query/hit，不能推出逐请求命中量。发起先后不能代替服务端入队/执行先后，缓存命中、路由、chunked prefill、系统状态、网络/客户端处理等均可能影响首条 SSE 延迟；现有资料无法区分。下一步只读检查原有请求与服务端日志是否保存非敏感请求位置→路由→入队/首次执行及 prefix 命中量的可靠关联与时钟口径；若无则标记缺失，不因相同序号推断内容身份或因果。该结果不修改服务器、不启动新负载。

现有本机 `raw-results/service-log-tail.txt`（2322 行）及两轮 `aisbench.log`（各 104 行）做关键字计数时，`request_id`、`cached_tokens`、`dp_rank`、`scheduled` 均为 0；service tail 亦无 `first token` / `first_token`。这是对已保存日志的有限检索，不排除其他命名、其他文件或未保存的服务器日志中存在关联；现阶段没有逐请求路由、缓存命中和入队/执行事件的可验证关联。

## 首条 SSE 的内容：完成接口的源码限定

两轮冻结 manifest 的 `effective_config` 均指定 `stream=true`、接口 `/v1/completions`，`generation_kwargs` 均仅含 `ignore_eos=true` 和 `temperature=0`，因此不能以 chat 接口的单独 role 帧解释这次结果。客户端 `VLLMPrefixCacheAPI` 的基类 `get_request_body` 合并这些参数并设置流式请求，没有主动设置 `echo`；该基类文件 SHA-256 `8170e1ca344340a4919e05fb61df7bfdcd8d7f7b7cca9d427924d793c89da086` 与 `benchmark-20261028/raw-results/client-source-SHA256.json` 对应条目一致。插件本身 SHA 为 `cb0424af9e1ad9dd694aff5e7a0249a31cff40d77ff37b5d8caadc23d04a266e`，同样与冻结清单一致。

已保存的 [当前容器源码快照](../bounds/completion-path-snapshot-status.json) 显示 completion 协议的 `echo` 默认是 `False`；正常非 echo 分支在此前尚无输出 token 时，若 `output.text` 与 `output.token_ids` 均为空则跳过该帧，然后才发送 `text_completion` 帧（[completion_serving.py](../bounds/completion-path-snapshot/completion_serving.py) 约 338–441 行；[completion_protocol.py](../bounds/completion-path-snapshot/completion_protocol.py) 约 57 行）。快照状态记录 baseline 与 stage 中该文件一致且目标路径无挂载覆盖；本次只读核对当前 baseline 文件 SHA：serving `75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527`，protocol `5f7930635fd3924826015211032fb298206b27c7f04cd6cd2224bb128e4a3a50`。**条件性结论**：如果两轮历史服务实际运行的是同一段源码、请求未被中间层改写且首帧为正常输出，上述代码使首个正常输出帧包含非空生成文本或非空 token ID 列表；它并不是 chat 的 role-only 帧。文本非空与“恰好一个 token”不同，服务只在请求 `return_token_ids` 时返回 token ID。

**仍不能证明历史首 token 时刻**：现有快照并非服务在两轮基线运行时的内存代码逐字节证明；日志和详情也未保存逐帧文本类别、帧中的 token 数或首 token 的服务端时间。对 16 个已保存详情作字段结构计数：每条仅有聚合 `prediction`、`output_tokens` 与指向数值数组的 `time_points`，`extra_perf_data` 和 `extra_details_data` 都为空；没有持久化逐帧内容与时间的关联。计时数组照 `recompute_baseline_latency_anonymized.py` 的唯一 DB 引用复算，不能由聚合结果反推出首帧 token 数。异常帧、实际请求改写及输出缓冲路径也未逐请求绑定，因此客户端所谓 TTFT 保持“发送到首条记录的 SSE data 行”的观测定义，不能严格当成引擎内部首 token 生成时间。上述检查只读，不输出请求或回答内容，不启动 NPU。

## 对照重采集的客户端首条数据：共同负载但不是同一计时器

只读 10-29 重采客户端保存的 `measure1-responses.json`（SHA-256 `ca431b9300f4907a4a05978019903d0bf58045fec1b9957b3637f24200701c06`），8/8 条具有有序的 start、first data、first nonempty text、HTTP end 单调时钟数值。程序解析整个已保存响应 JSON，但聚合**不提取或输出**请求、回复、原始 ID、SSE 文本或逐条身份。复算结果如下：

| 客户端范围，毫秒 | 非 profiler measure1 | 重采 measure1 |
|---|---:|---:|
| 八个请求 start 最大跨度 | 38.734 | 3.168 |
| 单请求 start → 首条已记录 SSE data：最小 / 中位 / 最大 | 300.687 / 921.729 / 1097.571 | 6260.582 / 7796.072 / 9617.801 |
| 单请求 start → HTTP 结束：最小 / 中位 / 最大 | 此处无同口径结果 | 43462.070 / 43724.726 / 44047.062 |

重采客户端 `diagnostic-20261029/profile_client.py`（SHA-256 `e3528495c48ae02f114dc3943263ba67c6494f795b67db4bfef9079408d9c2a3`）在发送请求前调用 `time.monotonic_ns()`；接收 `data:` 行后排除 `[DONE]`、完成 `json.loads`，**再**记录 `first_sse_data_monotonic_ns`；对第一个包含非空 `choice.text` 的 data 行记录 `first_nonempty_text_monotonic_ns`。8/8 保存的第一条 data 行都标记文本非空，两种时间在每条记录中相同。这仅表示该客户端拿到的首个 data 行含非空文本，不证明帧恰好一个 token，也不是服务端生成首 token 的时间。HTTP end 在 response 迭代退出后记录，不等于无 profiler 基线的最后非 DONE SSE 时间。无 profiler 客户端的 first 在读取非空、非注释、非 DONE 的 SSE 行后、JSON 解析**前**记录；两者的起点、网络实现及响应解析不完全相同。表格只陈述各自客户端观察值，不将其命名为服务端 TTFT 或直接相减求 profiler 开销。

负载一致性证据：重采 progress 报告的 `measure1_prompt_sha256` 等于对本机已保存 measure1 requests 文件中 8 个 question 按重采代码的 JSON 编码方式得到的列表摘要；这里只报告布尔匹配，不输出摘要或文本。历史 measure1 的配置引用同一冻结 requests 数据集；这仍不证明两个客户端实际 HTTP body 的所有字段、每个请求与服务端执行对象、缓存内容或设备工作完全相同。重采 progress 还记录 case_valid=true、8 个响应与冻结参考输出完全匹配，计数聚合层面 prefix query/hit 为 262144/235904、无抢占；这些不是逐请求 cache 命中或内核身份。重采不仅开启 profiler，还同时启用有同步日志写入的 hook，见 `RECAPTURE-SCOPE-REVIEW-20261029.md`；两次的服务进程、运行时间、缓存初态、客户端并发启动与环境也有可能不同。因此，数秒级客户端差异值得后续排查，但**不能量化或归因 profiler 单独增加了多少**，也不能把差异分配给 NPU、调度或网络。

对已有 `metadata-summary-early.json` 的匿名请求匹配计数，8 个期望客户端标识在 scheduler 和 runner 聚合中各匹配到 **0 个不同标识**。这是所存摘要和当时匹配方法的结果；ID 转换、密钥、采集覆盖等未闭合，既不证明这些请求没有被执行，也不能将任何一条客户端首数据直接绑定到某个 profiler 层或步骤。更准确的请求、阶段与设备轨迹关联仍需同次可验证的匿名 ID、时钟域和完整生命周期；若旧文件没有它们，就保留未知。

复核重采的脚本 [query_recapture_first_response.py](query_recapture_first_response.py) SHA-256 `7e4371075dcfc09cc5e0697ff33d40dd492c94f490955c1421235705af42776e`；只在自有 stage 容器读取上述响应文件，生成无逐请求身份与文本的 [recapture-first-response-anonymized.json](recapture-first-response-anonymized.json) SHA-256 `0250f204d00a021bb40c6c8a2e3866d27b305ff1d4d8b493a5e73acc5c57f451`。无 profiler 两轮的既有复核结果 `baseline-latency-anonymized.json` SHA-256 `0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9`。本次未触发 NPU 负载或变更推理实现。

## 无 profiler 基线：逐 SSE 行间隔与重叠

直接复算两轮每个请求中相邻非空、非注释、非 `[DONE]` 的 SSE 行记录时间差：不含 POST 到首行的首响应延迟，也不含末行到 HTTP 结束；时间记录在行解析前。每轮的 8 个请求均参加，输出均为 1024 token；每轮把这 8 个请求的行间隔合并，采用 NumPy 的 linear 插值计算分位数。

| 客户端行间隔 | 第 1 轮 | 第 2 轮 |
|---|---:|---:|
| 间隔数量 | 1759 | 1754 |
| P50 / P90 / P99 | 65.33 / 69.43 / 73.76 ms | 65.13 / 69.00 / 78.07 ms |
| 最大值 | 411.42 ms | 409.79 ms |
| 超过 100 ms 的间隔个数 | 11 | 11 |
| 存在超过 100 ms 间隔的请求 | 7 / 8 | 7 / 8 |

100 ms 是事后选定的描述阈值，不是服务等级界限。各轮有一个请求没有超过该阈值的间隔，**该请求没有被排除**；原因未知。按已存时间坐标，每轮都出现过 7 个请求各自处于一个大于 100 ms 的行间隔之内的时间点；这项跨请求重叠的物理解释以这些记录使用可比的同域时钟为条件，尚未独立校准。区间重叠不证明它们等待同一服务端阶段，也不说明设备同时闲置；客户端缓冲、接收处理、网络和服务端等都可能影响观察值。

行接收时间在 JSON 解析前记录，行内可能没有生成文本，也可能包含多个 token；不能把每行间隔叫作逐 token TPOT、引擎单步或算子时长。当前没有逐行文本和 token 数量绑定；上文对首帧的源码条件推导也不等于此处所有行的内容证明。合并间隔在请求内相关，分位数不是独立采样置信区间，只有两轮，不能推断其他并发和负载。不能从最大间隔、重叠或 P99 判定调度瓶颈、因果及收益。

复算（Mac 示例）`python3 agents/diagnosis/recheck_baseline_sse_gap_distribution.py --results-root /Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results | cmp - agents/diagnosis/baseline-sse-gap-distribution.json`；复制至其他环境时将 `--results-root` 指向同一冻结数据所在的 results 目录，哈希不匹配会拒绝运行。脚本先核对每轮 details 和 SQLite SHA 与既有基线摘要一致，再只提取必要计数和时间数组，以只读 SQLite、`allow_pickle=False` 读取；不输出文本、原始身份或逐条记录。脚本 SHA-256 `23a670a4ac446304bdf38789204ff2f1fbee4b78ff974f4a52182194935ade68`，聚合输出 SHA-256 `78c9587bb21edd7c5c47034d859aad07753103c005be540de7c25dac50cff8a6`。

### 长行间隔与最后一个首行的相对位置

复用上述两轮各八请求的同一客户端时间数组：每轮先取八条记录中最晚的**首个已记录 SSE data 行**时间，然后检查全部相邻已记录非 DONE 数据行之间的间隔。以事后选定的 `>100 ms` 为长间隔，两轮各 11 个长间隔**全部在该轮最晚首行之前结束**，跨过该边界的长间隔和从边界开始或之后开始的长间隔均为 0。最后一个长间隔的结束比最晚首行早约 0.049 / 0.051 ms，数值来自客户端时间戳差，不能扩大为事件时钟精度的独立保证。

另以间隔**起点在最晚首行或之后**作为子集，第一轮有 1738 个间隔、最大 73.783 ms；第二轮有 1733 个、最大 78.371 ms。该子集筛选不含跨边界间隔；上段对全部长间隔的检查单独涵盖了可能跨边界者。这里的时间数组按客户端规则排除空行、注释和 `[DONE]`，没有末条数据到 `[DONE]` 或 HTTP 完结的时间；没有保存的行或传输缓冲也不在范围内。

因此，在本次两轮记录与事后阈值下，可以排除“观察到的 >100 ms 相邻数据行间隔在所有请求收到首行之后仍反复出现”的描述；这不是逐 token TPOT 或服务端单步时间。首个 data 行是否包含恰好一个 token 未证，跨请求相对位置依赖这些数组属于可比的同一客户端时钟，且尚无服务端请求或阶段绑定。上述现象不能归因于 prefill、网络、调度或设备，也不能推广到其他负载；两轮及同轮请求并非独立随机实验。

只读复算（Mac 示例）`python3 agents/diagnosis/recheck_sse_gaps_after_all_first.py --results-root /Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results`，脚本 SHA-256 `2604ac7088c6f8792b18edb2c062e36735ed5ca78afd2345e73b13396b7a2401`。脚本核验固定的匿名基线摘要 SHA `0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9` 及其记录的两轮 details/SQLite SHA，检查唯一 DB 引用与时间单调，只输出匿名聚合；不读取设备或修改服务。
## 两轮基线的客户端未收完数据请求数

复用本文件先前核验的两轮各 8 个请求的客户端时间点。将每个请求定义为从“POST 前的客户端起点”到“收到该请求最后一条已记录、非 DONE 的 SSE 数据行”的半开区间；各轮窗口从最早起点到最晚一条数据。按同一客户端记录时间轴扫描这 8 个区间并计算时间覆盖：

| 指标 | 第 1 轮 | 第 2 轮 |
|---|---:|---:|
| 8 个请求均处于上述区间的时间 / 该轮窗口 | 96.35% | 92.13% |
| 按时间加权的区间内请求数 | 7.860 | 7.755 |
| 8 请求平均 POST 到最后数据时间 | 15.408 秒 | 15.350 秒 |
| 该轮 8192 token / 窗口 | 522.37 token/s | 517.35 token/s |

这说明两轮多数客户端测量窗口内，8 个请求都已开始且尚未收到各自最后一条已记录数据；**不能据此说 8 个请求在服务端均已入队、同时执行、都处于 decode 或占用 NPU**。客户端 POST、网络、排队和响应缓冲都可能包含在区间里，最后一条 SSE 数据也不是 HTTP 结束时间。上述结果只排除“这段客户端测量窗口的大部分时间里并没有同时存在 8 个上述客户端区间”这一描述，不能解释服务内部调度或瓶颈，也不保证其他负载和稳态表现。

平均区间数按 `sum(每请求区间时长)/窗口时长` 计算；对应 8192 token 的总吞吐可写成 `平均区间数×1024/平均请求区间时长`，这只是同一数据的恒等变形，不是吞吐上界或对未来请求的性能预测。不同请求、首 token 及各服务阶段的时间不能由此拆开。

复算：任务目录执行 `python3 agents/diagnosis/recheck_baseline_client_outstanding.py`。输入 `baseline-latency-anonymized.json` SHA-256 `0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9`；脚本 SHA-256 `2b209c04db16ebe06cc56d02299205c55962e58a504ad56ec9b0bb471d065535`，匿名聚合 `baseline-client-outstanding.json` SHA-256 `00e13faf57791dc354e9dd3fc7cce89f767598e6c749534ae91d22b74b50c71f`。不读取客户端文本，不访问设备或修改推理实现。

## SSE 行间隔和 TPOT 为何差约 4.6 倍

基线的每个请求都记录了 1024 输出 token；上面的客户端时间代理定义为 `TPOT=(末行时间−首行时间)/1023`。令同请求记录的非 DONE SSE 行数为 `n`，对同一段首行至末行时间，逐请求的平均行间隔是 `(末行时间−首行时间)/(n−1)`。因此两者之比**精确等于** `1023/(n−1)`：

| 逐请求指标 | 第 1 轮 | 第 2 轮 |
|---|---:|---:|
| 记录的 SSE 行数范围 | 218–224 | 210–227 |
| 平均行间隔范围 | 65.548–68.114 ms | 65.346–68.035 ms |
| 平均行间隔 / TPOT 范围 | 4.587–4.714 | 4.527–4.895 |

这解释了为什么观察到约 65 ms 的典型 SSE 行间隔与约 14 ms 的 TPOT 数值不同：它们用了不同分母，并不能将差值解释为设备调度、算子执行或额外等待。上表是逐请求算术平均行间隔；上文 P50 是合并所有间隔后的中位数，统计口径不同，不能直接用此恒等式逐项替换。

`1023/(n−1)` 也不是实测的每帧 token 数：缺少逐帧内容、token 身份与行之间的绑定，且可能包含空文本、聚合或特殊 token。仅凭这个比值不能断定服务端逐步生成或网络/客户端缓冲的具体方式，也不能排除它们造成额外延时。

复算直接读取先前校验哈希的 `baseline-latency-anonymized.json`：对每个 `phases.*.records` 中的 `first_response_offset_ms`、`last_response_offset_ms` 和 `recorded_non_done_sse_count` 按上述公式计算，并与记录的 `tpot_ms` 相除验证等式。两轮共 16 个请求的等式浮点误差最大约 `8.9e-16`；这复用旧数据，不是新采集，不检验历史 token 到 SSE 行的映射。

在任务目录复算以下匿名范围：

```sh
python3 - <<'PY'
import hashlib, json
p = 'agents/diagnosis/baseline-latency-anonymized.json'
raw = open(p, 'rb').read()
assert hashlib.sha256(raw).hexdigest() == '0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9'
for phase, data in json.loads(raw)['phases'].items():
    ratios, counts = [], []
    for row in data['records']:
        duration = row['last_response_offset_ms'] - row['first_response_offset_ms']
        n = row['recorded_non_done_sse_count']
        assert n > 1 and abs(duration / 1023 - row['tpot_ms']) < 1e-10
        ratios.append(1023 / (n - 1)); counts.append(n)
    print(phase, 'requests', len(ratios), 'line counts', min(counts), max(counts),
          'ratio range', min(ratios), max(ratios))
PY
```

## 无 profiler 基线的服务端阶段指标（2026-10-30）

两轮 `metrics-before/after.txt` 固定 SHA，离线复算 `python3 agents/diagnosis/recheck_baseline_phase_histograms.py`。每轮 `request_queue_time_seconds`、`request_prefill_time_seconds`、`request_decode_time_seconds`、`time_to_first_token_seconds` 和 `e2e_request_latency_seconds` 的 histogram count 都是 engine 0/1 各增加 4。queue 的 sum 两轮分别增加 0.00025722966529428954 / 0.0003420996945351362 秒，除以各自 8 次观察，均值约 0.032 / 0.043 ms。prefill 均值约 374.91 / 372.29 ms，decode 均值约 14.663 / 14.605 s，TTFT 均值约 750.01 / 749.23 ms。这些是服务端计数窗口中每次 histogram 观察的聚合均值，不是设备计算时长，也不是逐请求事件；不同 histogram 的观察是否一一对应尚未证明。

该任务 stage 容器当前 `/vllm-workspace/vllm/vllm/v1/metrics/stats.py` SHA-256 `af1983ddc32fa4d3284e72212c6525daf961dd84429023a02d612b68b3112cf3`，`:518–525,542–551` 显示 queued 为 scheduled_ts−queued_ts，prefill 为 first_token_ts−scheduled_ts，decode 为 last_token_ts−first_token_ts；preemption 在后两段可被包括。`loggers.py` SHA-256 `c12b4fbebfe886b697127f1114eaa0bb83c54a78675919d178b619c3d8d5d655`，`:1145–1155` 对 finished_request 的这些字段 observe。当前 stage 源码不是历史服务进程加载版本的证明。按该源码，queue 只覆盖首次调度前的引擎内队列间隔，不等于整个调度器 CPU、每轮迭代调度、预占恢复、网络或进入引擎前时间。

基线 `benchmark-20261028/raw-results/driver.py:46–60` 每轮发两条 seed、等 running/waiting 均为零后取 before，再执行 AISBench，返回后取 after；`:62–64,69–72` 记录服务成功及 token 增量和客户端 8/8 成功。每轮 between-scrape 数量与目标请求数相容，但 scrape 窗口不同于客户端计时窗口，且未按请求 ID 绑定。measure1 after 到 measure2 before 每引擎的完成 count 都增加 1，每引擎 prompt/generation 增量 29696/1，与 measure2 两条分别指定 DP0/DP1 的 seed 高度相容，仍不能仅凭聚合值断言其身份。

**条件性排除：**如果上述每轮 8 个 queue 观察恰为该轮 8 个客户端请求、各观察非负且历史语义与当前源码一致，那么任一请求该队列指标至多等于当轮总和 0.2573/0.3421 ms；它单独不足以解释客户端 300 ms 以上的首条 SSE 间隔。测量窗口和事件身份尚未绑定，不能将此升级为实际排队根因排除，也不能认为调度或网络开销可忽略。最小 bucket 为 0.3 秒；微秒级均值来自 sum/count 而非 bucket 分辨率。本节仅离线读历史快照及只读当前容器源码，无 NPU 负载。

### 同 cohort 和 TTFT 残差复核

`recheck_baseline_phase_histograms.py` 现校验每轮每 engine 4 条 generation histogram，`le=1000` 增量为 0、`le=2000` 增量为 4；同一轮每 engine preemption histogram count 增 4、sum 增 0。结果与每个已计观察输出超过 1000 token 相容，不能由桶断定每条恰为 1024。当前 stage `loggers.py:1138–1158,1167–1172` 在同一个 `finished_request` 循环依次 observe queue、preemption 与 generation，但对多个 histogram 的更新不是同一原子事件。如果历史代码与此一致、两个 scrape 均未穿越对应逐请求多指标更新，则这些 generation 观察与 queue 观察属于同一批，可排除输出 1 token 的 seed；仍无法区分其他长输出请求，也不能逐项绑定客户端。

各轮 `TTFT sum − queue sum − prefill sum` 为 3.0005440151 / 3.0151656175 秒，除以 8 为 375.068 / 376.896 ms。stage 的 `stats.py:224–230,427–453,467–473,497–499` 在前端的 `IterationStats` 用 `time.time()` 与 frontend arrival_time 计算 TTFT，engine core 的 queued/scheduled/first_token 使用 monotonic 时间戳；`loggers.py:1133–1134,1138–1149` 对 TTFT 在 first-token iteration 计数，其他两个指标在 finished-request 迭代计数。故残差受观察集合、计数窗口、前后端时间点、时钟速率和历史版本共同约束；尚不能把它称作某个独立阶段或算作调度/网络损失。当前阶段只记录可复算差值及待验证关系，不修改推理实现、也不增加设备负载。

### TTFT 残差的源码事件范围（当前 stage，条件推导）

只读取当前 stage 容器 11 个源码文件、核对其各自 SHA 后，截取限定行保存在 `agents/diagnosis/stage-ttft-source-slices.json`（SHA-256 `bea01b1729a5a53f05f1a23fead78f881b5e8d9d1e99328f5c2f5cd3753f6b92`）。在所审查的 completion 正常路径，`renderers/base.py:1130–1154` 的 `render_cmpl_async` 在渲染、分词及转为引擎输入之前记录 `arrival_time=time.time()`；此前 route 和 renderer 的预处理不包含在此起点内。`input_processor.py:309–313` 在没有显式 arrival 参数时使用引擎输入中的值；这是源码路径条件，不是历史调用轨迹证明。

core 的 `preprocess_add_request` 在交给 scheduler 之前构建 Request；如果启用 prefix caching 或 KV connector，`core.py:224–233` 会建立 block hasher，而 `request.py:219–224,280–281` 可在构造请求时更新块 hash。`scheduler.py:2485–2496` 入 waiting 后记 `QUEUED`。同一 step 的 `EngineCoreOutputs` 构造默认记 monotonic 时间（`scheduler.py:2255–2259`、`engine/__init__.py:279–281`），输出进入 frontend 后 `async_llm.py:785–805` 创建 `IterationStats` 并处理；`stats.py:427–453,467–473,497–499,542–551` 分别用 frontend wall clock 算 TTFT、core monotonic 算 queue 和 prefill。

令 A 为 renderer 的 wall arrival，Q 为核心 QUEUED，C 为该首个关联 core output 的 monotonic 构造时刻，F 为前端收到输出后 `IterationStats` 的 wall 时间。对**同一请求**、同一 first output、单次正确配对 QUEUED/SCHEDULED、当前代码版本确实生效且两种时钟都准确度量该段经过时间的条件下，`TTFT − queue − prefill = (F−A) − (C−Q)` 可在物理时间上解释为“renderer 起点到核心入队”与“核心输出构造到前端处理起点”两段经过时间之和，另有计时误差。这里右边是概念上的真实经过时间，不能直接用不同进程、不同钟域的绝对时间戳相减。该范围可能包含渲染/分词、请求构造和 prefix hash，以及进程间交接等；不包含 HTTP 到 renderer 起点之前或前端处理起点到客户端 SSE 之后的所有路径。代码没有给出两段各自耗时，也不能证明任何步骤是 375/377 ms 差值的根因。

当前 histogram 抓取没有逐请求 ID、事件与时钟关系，历史代码是否相同也未绑定；公式因此只是能被后续单次事件记录证伪的结构模型，不是该基线中测得的时间分解或前缀 hash 成本。本轮没有运行 NPU 或修改推理实现。

### 当前 stage 前缀 hash 路径和不可相加的反例

当前 stage `v1/core/kv_cache_utils.py` SHA-256 `2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2` 的所需源码片段保存在 `agents/diagnosis/stage-kv-hash-source.json`（SHA-256 `c0f6489fe4d44e5832a606d78f9096b421a255b8adc8b43c1d3cf49a68e7d378`）。`Request` 初始 `block_hashes=[]` 并调用 `update_block_hashes()`，仅 hasher 非空时执行。`get_request_block_hasher` 取完整块并将 `prev_block_hash_value` 传给下一块 hash；在首次构造、T 个 token、正整数 hash 块大小 H、没有之前 hash、对应路径确实启用的条件下，该源码循环执行 `floor(T/H)` 次。若 T=32768、H=128，这是 256 次/请求；数据跨请求、线程及 NPU 仍可能重叠，不能换算成设备耗时或整个服务的不可避免算法下界。

冻结启动命令 `--block-size 128` 并不足以单独证明历史 H=128。当前 `resolve_kv_cache_block_sizes` 单缓存组使用配置块大小乘 DCP，多组则可能按 prefix_match_unit 或可缓存组块大小的 GCD 推导；历史实际缓存组、DCP 与 resolved H 尚未绑定。哈希发生在所存的 Request 构造、调度器入队之前；如果代码及条件适用，已有前缀缓存命中不自动免除该构造阶段为完整输入建立块 hash，但不能从 90% 命中率推出实际 hash 时间或者 375/377 ms 差的原因。若为了减少该路径开销进行跨请求复用或重排，应保留每块前继 hash、token、extra keys、模型/盐以及跨请求生命周期；也不可从现有串行实现得出所有语义等价方案均必须串行的严格界。

两轮基线 `E2E histogram sum − TTFT histogram sum − decode histogram sum` 分别为 −0.05960817309096 / −0.05674831848591 秒，除以每轮 8 次观察分别 −7.451 / −7.094 ms；复算 `python3 agents/diagnosis/recheck_baseline_phase_histograms.py`。TTFT 和 E2E 是前端墙钟指标，decode 是 core monotonic 首末输出区间，且 TTFT 与 finished hist 在不同事件计数；即使计数一致也没有请求配对。这个负值否定把这些聚合当作同一组、边界完全衔接的互不重叠正时间阶段后再做加减的简单解释，不证明记录出错或能归因于具体因素；不能从中拆出 CPU、NPU、网络或调度时间。

### 已保存启动日志对哈希粒度的条件约束

`python3 agents/diagnosis/recheck_prefix_hash_configuration.py` 校验固定启动日志和启动脚本 SHA 后，仅输出非敏感聚合：DP0/DP1 各记录 12 个 KV group，大小均为 128，LCM 为 128；同一启动日志的配置摘要记录 DCP=1、prefix caching=True。冻结 `frozen-serve.sh` 指定 `--block-size 128`，未显式指定 `--prefix-match-unit` 或 `--prefix-caching-hash-algo`。与当前 stage 的 `resolve_kv_cache_block_sizes` 相结合，如果启动记录对应本次服务、历史运行使用同一解析源码、match unit 没有后续或其他覆盖、且 hasher 初始化实际生效，那么组大小的 GCD 为 128，可推导 H=128 和初始 32768 token → 256 个完整块。这增强了 256 的配置相容性，仍不是历史 resolved H、具体 hash 函数或实际物理计算次数的直接观测。

当前 stage `hash_block_tokens` 注释声称 LRU，但所查定义前没有缓存装饰器，函数体将 token 转 tuple 并调用传入的 hash 函数；源码范围内静态调用没有显示其他包装。运行时补丁、传入函数内部及历史对象未绑定，不能据此证明缓存未发生、256 次物理 SHA-256 或相应耗时，也不把这些实现工作作为允许缓存和替代算法的理论下界。原始日志未公开；复算脚本只输出聚合字段。

### 客户端启动顺序与首条 SSE 延迟的配对描述

复算 `python3 agents/diagnosis/recheck_client_first_response_order.py`：使用匿名时间数据 SHA `0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9`，每轮按 POST 前 start 排序，首条已记录 SSE 延迟的次序名次分别为 `[1,3,2,7,5,6,4,8]` / `[1,3,2,7,6,5,4,8]`，Spearman 分别为 0.7619 / 0.7381。按事后划分前 4 / 后 4，第一轮平均 568.877 / 970.861 ms，第二轮 570.434 / 978.852 ms；全轮客户端启动跨度 38.734 / 39.215 ms。每个延迟定义为该请求 `first_response_offset_ms−start_offset_ms`，因此不能将起始错开时间另加作为这约 402 / 408 ms 组间差异的解释。

这是两轮各 8 条且同一次请求内配对的描述；请求索引和启动顺序不能代替服务端接收、排队、运行、DP 路由顺序或真实 prefix cache 命中，输入与有效计算量也未控制，无法得出启动次序造成变慢或统计长期规律。两轮来自相似的固定测试，不能当独立试验确定因果、置信区间或可达提速。仅使用现有匿名时间点，没有设备负载。

### 按 engine 标签对照草稿计数与 decode 指标

对两轮各自的固定 `metrics-before.txt` / `metrics-after.txt`，按同一 `engine` 标签分别求计数器增量，并与同窗口的 `request_decode_time_seconds` 直方图增量并列。接受比例是该标签下 `accepted_tokens / draft_tokens` 的聚合比值，不是每个请求的平均接受率。decode 均值是对应直方图的 `sum` 增量除以 `count` 增量；每组 count 均为 4，生成 token 计数增量均为 4096。

| 轮次 | engine 标签 | 草稿次数 | 提议 / 接受 token | 接受 / 提议 | decode 均值 |
|---|---:|---:|---:|---:|---:|
| measure1 | 0 | 881 | 4405 / 3219 | 73.08% | 14.683 s |
| measure1 | 1 | 870 | 4350 / 3228 | 74.21% | 14.642 s |
| measure2 | 0 | 864 | 4320 / 3241 | 75.02% | 14.385 s |
| measure2 | 1 | 882 | 4410 / 3215 | 72.90% | 14.824 s |

每轮 decode 均值较高的标签恰好也报告了较多草稿次数；标签方向在两轮之间反转。这只描述四个聚合组，不能认定更多草稿或接受率造成较长 decode。草稿计数不是 NPU 执行步数，两个指标的更新时点不同，抓取不是跨指标原子操作，也没有逐请求的 engine、草稿与 decode 事件配对。生成 token 总和为 4096 只能约束计数窗口内的总量，不能证明每次观察的长度恰为 1024，也不能把这四组视为独立受控实验。

离线复算：`python3 agents/diagnosis/recheck_engine_spec_decode.py`；如果数据不在默认位置，追加参数指定包含 `measure1`、`measure2` 的 results 目录。脚本核对四份快照 SHA-256，与 `recheck_baseline_phase_histograms.py` 的固定值一致，检查每个标签的计数增量及非负条件，使用 Decimal 计算并只输出匿名聚合。脚本 SHA-256 `81a862de09576e06011b3efd87e8b8f1d925b849a9e69003755e9b548a67a522`。这是既有记录的只读 CPU 复算，没有运行 NPU 或修改推理实现。


## 两轮无 profiler 基线的 engine 标签聚合

从同一轮 metrics-before/after.txt 的 histogram 各按 engine 标签作差：TTFT、prefill、decode 各标签的每轮 count 均增加 4。各自 sum_delta/count_delta 得到以下**不同 histogram 的组内均值**：

| 轮次 | engine 0/1 TTFT ms | engine 0/1 prefill ms | engine 0/1 decode s |
|---|---:|---:|---:|
| measure1 | 860.469 / 639.561 | 420.364 / 329.466 | 14.683 / 14.642 |
| measure2 | 858.479 / 639.981 | 416.637 / 327.946 | 14.385 / 14.824 |

TTFT 差值分别约 220.909/218.498 ms；两轮的 TTFT 分桶增量相同：≤0.5 s 为 0/2、≤0.75 s 为 1/2、≤1 s 为 3/4、≤2.5 s 为 4/4（每格为 engine 0/1 的累计 count）。这排除了“两个标签在这两个观测窗内各自的 TTFT 均值和这些桶增量均相同”这一有限描述；decode 均值高低方向两轮反转。每标签每轮仅 4 条 observation，指标没有绑定逐请求 ID、具体运行时设备和请求的实际工作量；engine 标签不自动等同某张芯片。不能据此断言某引擎或硬件较慢、客户端前后四条映射到哪一组、或用跨 histogram 的差值解释 TTFT 根因。输出 token 的各组均值为 1024，但总和除 count 不能证明每条观测恰为 1024。

复算 `python3 agents/diagnosis/recheck_baseline_engine_histograms.py`，脚本 SHA-256 `6bfbffe4f231f7cd617706c336ca2c70f14084d55adc6a97e0ec10729284bbe9`；它复用已有验证脚本中四份固定 SHA 的 scrape，只做同标签的 counter 前后差，不读请求内容、不启动设备。去标识后的组内统计与独立算术脚本在公开报告目录，不能替代逐请求绑定。


### 同 scrape 窗口中的 prompt 与缓存聚合

同一两轮窗口，每个 engine 的 prompt histogram count 均为 4、sum 均为 **131072**；缓存查询增量各为 131072，命中为 engine 0/1 **118016/117888**，本地计算 token 的计数增量为 **13056/13184**，两轮数字相同。每个 engine 的 prefill-KV-computed histogram count 为 4，其 sum 恰与对应 local_compute 增量一致。这排除在同窗聚合计数口径下“TTFT 均值较高的标签 0 报告更多未缓存 prompt token”这一描述；若这些数据确属同一批请求，则聚合总量更多也不足以解释该差异。

TTFT 在首 token 与 prompt/prefill 相关指标在其他事件时点更新；scrape 非跨指标原子快照，等 count 和相同窗口不证明相同请求集合。即使两批对象一致，请求内未命中分布、chunk 顺序、prefix hash、调度、前端/设备耗时和缓存的实际动作仍可能不同。已有客户端的请求 ID 尚未映射到 engine 标签，不能将组均值当逐请求差异或断言 TTFT 由缓存导致。最小可补证为同轮匿名 request→engine→实际 prompt/cache/computed→阶段事件及时间域的关联，再核对指标版本和进程身份；此阶段未发请求或新增采集。

扩展后的复算脚本 SHA-256 `6bfbffe4f231f7cd617706c336ca2c70f14084d55adc6a97e0ec10729284bbe9`；公开匿名数据提供聚合数字而非原始请求。
