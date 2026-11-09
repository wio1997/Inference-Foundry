# 文本到SSE交付与长prompt预热判断

## 介绍

从实际FastIncrementalDetokenizer到stop-string、logprobs、RequestOutput/collector、completion SSE，再到Starlette UTF8和Uvicorn HTTP写入，先看整段可删计算/交付。冻结C8的完整frontend trace是已有证据，不新增服务请求或profile。每个替换原语逐位一致。

## 到了哪里

运行树13源和6个ASGI/安装依赖已pin；当前源身份不代表当时加载字节码。437834个完整Python事件ID唯一，437797条存在的父边同thread且区间包含，37个root/missing；父子并集去重。self remainder仍可能含抢占/C-extension/未记录等待，不宣称是真CPU占用。Chrome baseTimeNanoseconds加ts才是wall clock；8个client offset仅差670ns。

调用路径：EngineCore消息→output_handler分块→process_outputs→fast update逐ID解码/stop holdback→delta CompletionOutput→RequestOutputCollector→merge_async_iterators→completion_stream_generator→Pydantic JSON字符串→Starlette UTF8→Uvicorn HTTP chunk/transport.write。collector已支持积压delta合并；此窗没有RequestOutput.add调用，不虚构可删完整历史重算。8请求8192 native step，1762次请求输出/流式文本DTO；JSON1770次包含最后usage，另有DONE/HTTP结束，不等于8192次重解码历史。

保留：min_tokens后stop earliest-completing-end/list-order、include/exclude截断、最长stop−1 holdback/finished释放；特殊token spacing、None/UTF8不完整和InvalidPrefix reset/其它错误；delta IDs/logprobs空切片、parent n/FINAL_ONLY/stream_interval；echo/max_tokens0/return_ids/usage/finish/error/DONE；JSON显式None和exclude_unset差异；disconnect/cancellation/backpressure/HTTP framing及transport存储所有权。当前固定请求logprobs/prompt_logprobs未启用，guard调用不代表重概率物化。keepalive仅正有限interval启用，trace无其异步循环；流控drain未见完整Python调用，但新实现仍须守卫其合法分支。

## 结果

**待确认，值得深入：长prompt原始seed解码。** installed tokenizers0.22.2，模型decoder为ByteLevel。8个约10ms的大native step合计84.266439ms；源码初次step先decode(prompt)，再decode(prompt+next)，恰好有8个构造，但trace没有参数/可靠request关联，不能直接声称每个大call归属。native全免的宽松固定schedule敏感性最高84.266439ms（P90 TTFT约5.86%，整批0.482%）；不能因整批不足1%剪去首token有价值的机制。包含所有构造的更宽oracle为88.658664ms/6.17049%，保留原构造验证的方案不能认领这4.392ms。rho=.5且每seed付.1ms时native条件净41.33322ms，非预计或实测请求收益；实际关键路径暴露仍缺。

上游v0.22.2 Rust源码已保存hash，揭示内部ids/prefix/prefix_index状态；Python ABI没有clone/prefix setter。它不证明已装binary来源。下一Loop定点验证：仅不可变ByteLevel且最后有效token以ASCII字节结束时，用空seed替代prompt seed；先调用原构造保留完整ID转换错误，再选择新stream。未完成UTF8/无安全边界回退，不能套用slow decoder五token lookbehind。比较每一步字节/None/异常及后续状态，不能只比较最终拼接文本。

**已停止：同C8独立1%预算的JSON-only小方案。** JSON全部免费也仅14.875935ms/整批0.0851%，所有首token之前宽松P90信用仅0.255192ms/0.0178%。不再同预算微调JSON/字符串拼接；不剪长prompt seed或ordinary独立负载。整个completion_generator约134.820ms、首token敏感性19.670ms包含dispatch/合并/恢复，并非JSON的预算。

**待确认：整output/BatchFrame/下一轮submit。** 选中父callee区间并集609.977455ms（整批3.489%）、放松ownership/到达后的P90信用113.260673ms；保留metrics、错误、engine生产等待及新lease开销，不能把609ms当预期省时。不同scope彼此重叠，不相加。

**待确认：输入renderer/tokenization。** 8个encode_batch在后台线程30599串行，共466.209239ms；不是主event-loop阻塞证据。需追真实renderer队列、同负载输入IDs/特殊token和BPE边界，才能判断批处理/前缀共享/并行代价。先完成上面改变去留的seed点证，再扫输入caller，四个warmup helper内部及startup MM budget仍未扫完。

## 证据与限制

`frontend-output-{frontier,execution-identity,extra-dependencies,trace-receipt}.json`；完整压缩事件本地`frontend-output-events.json.zlib`；`tools/analyze_frontend_output_frontier.py`。具体scope价格/有成本情景见frontier。8请求P90按nearest-rank取max，profiled整批17480.95198ms、P90 TTFT1436.816992ms；与reference arm不能做替换前后对。所有价格固定原arrival/engine schedule并放松CPU暴露和request归属，既非紧因果上界也非保证收益。

本Loop新增NPU/模型权重/服务请求/profile/远端写入均0；只读取已有源/trace和CPU库ABI，未改服务。范围是此交付的限定源码/估价判断，不是全链路完成或候选位一致准入。
