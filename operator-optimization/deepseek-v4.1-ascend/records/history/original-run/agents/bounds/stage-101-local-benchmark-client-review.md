# Stage 101：本机客户端源码及历史哈希核对

## 核验与适用范围

`benchmark-client-snapshot-status.json` 保留了 stage 101 的 `STOP_MISSING_OR_UNREADABLE_FILE: prefix_api`：stage 容器内指定的 `/data/.../client-packages/` 不可见，未生成远端快照。本轮没有修改挂载、环境或重新访问服务器。

本机 `/Users/wio/work/173-sglang-matched-172-benchmark/package/tools/aisbench-7bcb6440da11/` 中下列文件 SHA 与基线保存的 `benchmark-20261028/raw-results/client-source-SHA256.json` 对应项逐一相同。前四项是上轮指定目标，后四项只沿已发现的实际继承、结果路径核对：

| 相对路径 | SHA-256 |
|---|---|
| `plugins/prefix_cache/ais_bench_prefix_cache/models/vllm_prefix_cache_api.py` | `cb0424af9e1ad9dd694aff5e7a0249a31cff40d77ff37b5d8caadc23d04a266e` |
| `plugins/prefix_cache/ais_bench_prefix_cache/openicl/icl_inferencer/prefix_cache_gen_inferencer.py` | `adab7bf8b7b4042e454b7867863a87a254a9c429b74fca36eb6459e61ec05a0d` |
| `ais_bench/benchmark/models/api_models/vllm_custom_api.py` | `8170e1ca344340a4919e05fb61df7bfdcd8d7f7b7cca9d427924d793c89da086` |
| `ais_bench/benchmark/calculators/default_perf_metric_calculator.py` | `d7411d3ba02ab530ecd733b48c49f067ab6c92e1179d2054316487012c51cdd6` |
| `ais_bench/benchmark/models/api_models/base_api.py` | `203a416444b972193274a13e1593328a288b6879a3d62e4f201bbb9adfa81755` |
| `ais_bench/benchmark/models/output.py` | `ecf7600edf599cae33f8024fac0d2d1b285050110b4a45d7f6c6ec0d03c4836d` |
| `ais_bench/benchmark/summarizers/default_perf.py` | `e33857dd2a55030bd594d554d30598030fc4358489dc539ebe50dd54b3a2fd43` |
| `ais_bench/benchmark/openicl/icl_inferencer/output_handler/gen_inferencer_output_handler.py` | `d5832ed1e192656d1cffbfb483b8466f2da23c99c081d4d332cb53b66586ab38` |

基线 `launch_formal_benchmark.py:22,42-44` 配置客户端 `PYTHONPATH` 为任务 adapter 与 client-packages 并在目标容器中运行 `driver.py`，driver 在正式阶段调用 AISBench，配置为 `VLLMPrefixCacheAPI`。这些资料与历史源码哈希一起支持使用上述实现；不能以哈希单独证明历史进程逐次实际 import 或不存在 runtime monkeypatch。包的本机路径仅用于源码核查，不涉及 172 主机操作。

## 正式 payload 及结构化输出

- `base_api.py:108-114` 将配置中的 generation_kwargs 复制，只移除 response anomaly 的内部开关。冻结 `measure1.py`、`measure2.py` 给出生成参数仅 `ignore_eos` 和 `temperature`。
- `vllm_custom_api.py:114-138` 构造 `prompt`, `stream`、上述生成参数、`max_tokens`、`model`；流式时再加 `stream_options.include_usage=True`。
- `vllm_prefix_cache_api.py:61-93,138-151` 从父类取得 body，仅加入内部 DP 路由标记并在实际 POST 前移除，用 HTTP header 表示 DP。所核验构造链不插入 structured-output/response-format/guided 字段。**条件结论**：配置未被更改、正常执行这些已哈希源码且无额外运行时改写时，本次正式请求 payload 不含这类字段；这比“测量配置没有显式字段”更强，但不是对服务器端所有默认设置或每个历史 request 对象的直接观测。
- service log 的 `StructuredOutputsConfig` 是服务级配置，不能单独等同于请求启用。`request.use_structured_output` 如何从未携带字段的服务端请求算出，当前未快照，严格对象级结论仍待服务端相应源码或历史非敏感标志数据。

## 客户端 token 与计时

- `vllm_custom_api.py:140-143,179-185` 的流式响应解析读服务返回的 `usage.completion_tokens` 到 `output.output_tokens`；`vllm_prefix_cache_api.py:151-186` 按 SSE 分块循环调用该解析。`output.py:20` 初始化为 0，结果 handler `gen_inferencer_output_handler.py:76` 传递该字段。`default_perf.py:161-162` 仅当字段**不存在或为 None**时用 tokenizer 对预测文本重算；字段为 0 时并不会用这条分支重算。因此按已核验正常路径，最终正值 1024 来自解析到的服务端 usage，非该 summarizer 的 tokenizer fallback。不能据此断定所有下游数据从未被其他未检查代码或运行时替换更改。
- `default_perf_metric_calculator.py:80-81` 用本阶段所有请求 end 时间最大值减 start 时间最小值。`vllm_prefix_cache_api.py:150,167-174` 在发 POST 前记录起点，并在每个非空且非 [DONE] 的数据块解析前记录时间点。该时间不等于每个 engine 提交事件的窗口；服务端的 `generation_tokens_total` 增量也是 metrics 前后独立窗口。
- 两轮服务计数总量及客户端计数均为 8192，是一致性证据；`agents/diagnosis/spec-counter-accounting.json` 记录的 accepted+drafts 分别为 8198 和 8202，反驳将它们直接等同于生成数。逐事件映射、绝对时间边界和差异原因仍未知。

## 尚缺证据

要将上述条件变成所有历史请求上的严格事实，至少还须服务端 API request 到 `request.use_structured_output` 的默认值与映射、历史进程实际导入/改写情况，以及客户端 usage 与 scheduler 的 `EngineCoreOutput.new_token_ids` 的准确映射。已经没有必要为寻找这四份客户端源码修改容器或再次访问服务器。没有证明这些前，不应将 5+1 推断或计数平均值当全局物理 TPS 上界。
