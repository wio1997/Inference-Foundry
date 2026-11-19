# D侧准入候选：资源快照、条件容量和计时语义修正

## 结论

[正式32K/64K的阶段直方图](056-formal-stage-snapshots.md)支持优先调查SG P/D bootstrap及D侧`kv_transfer_alloc_ms`命名区间，但**尚不能断言full-KV容量是根因**。这次核对同轮次前后资源gauge后，排除一种错误解释：**不能只看到free少于一个长上下文，就说池已无可用容量**；同一个metrics响应还报告了大量可驱逐缓存。与此同时，客户端C16在“16条互不共享的完整上下文同时驻留同一D池”的条件假设下超过池容量，使容量相关准入值得继续验证，但这不是观测到的实际驻留状态。

G181的D启动记录配置Ascend传输、`early`分配策略及decode radix cache。G316独立256K诊断也记录early策略。候选实现说明，`bootstrap_done`是调度器首次poll看到接收侧`WaitingForInput`的时刻；**不能再把18.060ms子指标笼统称为已证远端网络握手或wire时间**。G181计时源码、G181/G316接收端连接实现均未逐文件绑定到现场，候选路径不证明运行时命中或数据面完成。

## 32K/64K正式轮次前后：D rank0读数

以下是metrics响应时读取的已发布gauge，不是压测中间的实时资源时间序列；指标可能按scheduler周期发布，候选实现的空闲发布存在限速，不能保证各gauge在采集时刻同步更新。`available`、`evictable`、`used`的含义取自exporter自描述：空槽、可驱逐缓存、活动占用；并不证明待准入请求当时能回收这些槽。

|轮次端点|pool max|available|evictable|used|reported running / D prealloc queue|
|---|---:|---:|---:|---:|---:|
|32K cold before|345856|8448|337408|0|0 / 0|
|32K cold after|345856|8448|337408|0|0 / 0|
|32K hot before|345856|8448|337408|0|0 / 0|
|32K hot after|345856|8704|337152|0|0 / 0|
|64K cold before|345856|13312|332544|0|0 / 0|
|64K cold after|345856|13568|332288|0|0 / 0|
|64K hot before|345856|13568|332288|0|0 / 0|
|64K hot after|345856|13312|332544|0|0 / 0|

各行数值 `available + evictable + used = 345856` 只是已发布数的加法；**不能将available+evictable直接当成请求的allocatable预算**，因为锁定前缀、保留、正执行/撤回、SWA、Mamba、metadata和请求槽等都可能影响准入。八个端点报告的running与prealloc均为0，不证明整个窗口未排队或未耗尽。P对应池max为292864，各端点available 6144、evictable 286720、used 0；同样只是边界发布值。

## 极简容量算术：有条件，不是实测并发

D已归档启动信息记录 `max_total_num_tokens=345856`、`max_running_requests=16`、`page_size=128`、`num_reserved_decode_tokens=512`、启用decode radix cache；这是历史启动信息，不是对每个请求入队瞬间的回读。

|输入长度|假设16条完整、无共享输入同时占用该池，仅输入的需求|池容量 ÷ 完整输入的整数上限|若每条还需假设性预留512槽，16条需求|
|---|---:|---:|---:|
|32K|524288 token槽|10条|532480 token槽|
|64K|1048576 token槽|5条|1056768 token槽|

这些是假设所有请求同时驻留、各请求无prefix共享、输入全部计入同一已报告容量池的反事实；预留列额外假设每请求均在该池真实占用512槽。实际请求是否同时到D、完整KV何时分配、输出/预留及SWA对应哪个池都未知。C16只说明客户端并发上限，不代表D侧恰好16条全量驻留；每池的page/保留、实际可驱逐、锁和其他门槛须以同版运行时证据确认。按可驱逐gauge简单相加能覆盖单个32K/64K输入是端点算术，不证明该时刻真实准入，也不推翻运行中容量不足的可能。

## 候选调用链：为什么不能叫纯握手或单次malloc

以本机G156/G32候选源码为例（**非G181现场逐文件SHA证明**）：

1. `early`策略在receiver完成bootstrap信息获取/复用及KV参数注册后，把状态设为`WaitingForInput`并返回，未进入后续显式发送 `BOOTSTRAP` 并等待prefill完成的分支；前面的route信息获取和注册本身仍可能涉及网络。Ascend receiver经Mooncake继承该公共receiver。启动参数不证明现场上述每个文件和分支已命中。
2. D scheduler poll发现 `WaitingForInput` 后首次设置`bootstrap_done`；后续调度还检查metadata槽、请求槽、Mamba、可能的LoRA/HiSparse条件与full/SWA预算。候选radix路径会匹配并锁prefix后重新算预算，可驱逐值并非自动等于可立即准入。
3. 候选D计时的 `kv_transfer_alloc_ms` 是 `transfer_entry − bootstrap_done` 的时间差；可能包含排队、准入判定、分配、调度或提交，而不是单次allocator调用耗时。P的同名指标走不同分支，不应按名称视作相同语义。某些候选异步路径可在设备event就绪或metadata真正发布前记录transfer-entry，因此也不能从该值证明全部metadata或KV传输已完成。
4. 候选初始化在某些缓存路径可能发生在prealloc-entry计时之前；时间戳首次写入、重试/retract复用或zero/陈旧时间均需逐事件核实。尚无证据证明这些异常实际发生。

候选源码指纹和检查范围：`req_time_stats.py:1116–1147` SHA `21e17d68d3f18ffdb1a1203fe814f2ddcea9869021a3fa9f43210222d114a1e6`；`decode.py:1206–1239,1528–1663,1728–1749` SHA `39cf972b6bcc04cc8c150c650d845daaa68ccb511a69a71c5b0b8bdf7322a70d`；`common/conn.py:2556–2613,2658–2713` SHA `b89eb8fb2ec137d8229094e3762f73f8f1c675e067187c2cebd7996af92f96bf`；`metrics_reporter.py:1472–1498` SHA `29de9f39c6624deceaea5cd40a3f9a0828aadf8601eda3f64b8fd6d1404508d2`。这些是本机候选源码指纹，不含现场连接标识或请求。G316后验已核验的`req_time_stats.py`和`decode.py`可支持对应setter和poll语义，但它们不证明G181同版，也不证明G316 receiver连接源码同版。[256K诊断记录](033-256k-diagnostic-stage-evidence.md)保留18.060ms和1102.512秒的历史条件性数值；计数相同不证明样本相同，无重置和未跨setter更新仍是必要条件。**更正：曾把18.060ms写成“真正握手”证据不足，应读作本地prealloc-entry到首次观察`WaitingForInput`的指标子段，不等于远端握手或纯网络wire。**

## 最低成本的证伪

先检查已归档、已授权且脱敏的资源和源码身份；本地已检查的正式轮目录只有边界快照，没有同目录内落在正式before→after之间的其他采样。未搜索到中间快照不等于历史现场绝无数据。若要识别单次长间隔，需要在同一请求、同一服务epoch记录时间戳先后、poll/队首位置、实际full/SWA/evictable/锁定及保留预算、metadata/请求槽/Mamba条件，并区分转移入口与异步metadata/KV完成。若某长间隔内队首全部资源门均满足、无前项阻塞且调度循环持续运行，则简单容量不足不能解释该段；若时间戳顺序或重入状态异常，也不能把这段当真实等待。

本次没有提交新模型请求或更改服务；新增埋点、受控负载或A/B必须先获得独立授权。不同框架/不同窗口的stage均值不能相减作为TTFT根因。

## 复核

[匿名端点和条件算术](../evidence/kv_endpoints_057/summary_v4.json) SHA `e077f2e812d55add63ff9098e4a4b49d1821a916fc2c2913ce18431dd008ece7`。输入为[历史D启动信息](../evidence/matched_SG_G181/p/functional_PD_g182/startup_d_server_info.json)与32K/64K各轮前后快照；脚本只提取固定字段、rank0、角色及原始SHA，不输出请求、响应正文、标签值或绝对采样时间。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_kv_endpoints_057.py --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_summarize_kv_endpoints_057.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```
