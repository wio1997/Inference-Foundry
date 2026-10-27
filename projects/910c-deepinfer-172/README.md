# 172 · DeepInfer / DeepSeek V4.1 Flash W8A8

**固定重复文本热身方案：815.25ms平均首token、1312.32token/s，16/16完整输出一致。** 冻结vLLM为831.25ms/518.42token/s。

结果包含额外的输入分词LRU缓存；实际模型KV命中仍为89.990234375%。无此输入缓存的v20为1013.35ms，首次新文本仍付BPE成本；不能把815ms外推到新文本或任意负载。两轮平均领先16ms，见[差距排查/源码/原始结果](optimizations/20261027-gap-closure/README.md)。

当前真实模型worker为v20，HTTP为v21；当前记录优先于下方v8/v12历史现场。[实际配置](optimizations/20261027-gap-closure/runtime.json) · [状态](STATUS.json) · [上一轮v10–v13](optimizations/20261027-ttft/README.md)。

## v8 冻结结果（此前版本）

**固定吞吐目标达标：668.35 token/s，对比已冻结 vLLM-Ascend 518.42 token/s，提升28.92%。**
16/16正式请求成功，每个完整1024-token输出文本与冻结vLLM预测逐字节一致。
本目录独立记录适配，不覆盖[原vLLM基线](../910c-deepseek-v4.1-flash/runs/tuning-20261026T211228/TUNED-BASELINE.md)。

| 正式轮次 | DeepInfer 输出token/s | vLLM 输出token/s | DeepInfer TTFT ms | vLLM TTFT ms | DeepInfer TPOT ms |
|---|---:|---:|---:|---:|---:|
| 第一轮 |716.50|509.52|5502.0|885.7|4.6|
| 第二轮 |626.26|527.63|5167.2|776.8|6.6|
| 合并 |668.35|518.42|—|—|—|

同一172服务器、8张物理910C卡/16个逻辑设备，DP2×TP8、EP16；同一只读ModelSlim W8A8权重、tokenizer和冻结输入。
每轮8请求、32768输入/1024输出、并发8、greedy、ignore_eos；每轮输入262144、输出8192、实际缓存命中235904（89.990234375%）。
合并吞吐按 `16384 / (11.4333953 + 13.0808413)` 计算，不取两轮吞吐的算术平均。
两轮范围626.26–716.50 token/s，相对算术平均的极差13.44%；仅做用户指定的两轮，较慢一轮也超过目标。

**限制：DeepInfer首token明显更慢。** 本次达标指总输出吞吐，不代表所有延迟指标达到vLLM。
投机验证以块返回token，SSE到达节奏与逐token解码不同，TPOT不能单独代表均匀的实时输出速度。
这组重复上下文利于提示查找；结果不外推到任意输入、随机采样、通用连续批处理或通用模型准确率。

## 实际实现

从170已有源码快照继承910C适配基础，在172实现完整40层原生W8A8加载、TP8注意力、全局EP16 MC2、INT8 grouped matmul、
C1/C2压缩、索引/MLA、物理SWA页缓存、跨两DP副本调度、前缀快照和HTTP/SSE接口。
当前性能路径为 **输入提示n-gram查找（8/4阶，最多63草稿token）+64-query完整目标模型图验证**，不是DSpark/MTP。
每个输出token均经完整目标模型验证；拒绝时恢复长度、CPU Engram历史、C2待配对状态并做原生修复。
引擎不读取保存的vLLM预测；预测仅由独立基准客户端用于验收。

依赖固定vLLM-Ascend镜像内的编译CANN扩展/custom OPP，并复用CPU tokenizer；不启动、代理或调用vLLM Python引擎。
仅Engram表在CPU，主干与专家权重常驻NPU，不开启主干CPU prefetch。CPU侧通过固定vocabulary lookup和只读mmap选行减少开销。
当前MC2独立HCCL组1024MiB、预填充分块4096；v8冻结配置为2560MiB/512；图覆盖40层目标、索引/注意力、缓存写入和MC2。历史native基线59.40、138.83 token/s均保留。

## 当前服务与文件

- 目标仅172，`172.16.10.172`；监听`127.0.0.1:8172`，model=`dsv41`。
- 项目：`/data/tiankuan/wio/910c-deepinfer-172`；权重：`/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8`，只读。
- 当前native控制目录：`serving/prefill-tail-ffn-v20`（worker v20 / API v21）；服务冻结时健康，保留运行。
- 原`wio-dsv41-vllm-s172`已停止但保留，原基线数据/配置未覆盖。
- [当前运行配置](optimizations/20261027-gap-closure/runtime.json)、[当前结果](optimizations/20261027-gap-closure/current-result.json)、[v8冻结配置](config/runtime.json)、[状态](STATUS.json)。
- [v8复现与恢复](REPRODUCE.md)、[源码引用](source/refs.json)、[源码补丁](source/deepinfer-910c-w8a8.patch)。
- [原始证据包与SHA](evidence/archives.json)、[失败记录](FAILURES.md)、[独立源码状态说明](source/ascend910c-status.md)。

镜像为`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，固定本地image ID见配置；无registry RepoDigest，不能把image ID当可拉取的registry digest。
NPU容器本次privileged已获明确授权；容器根目录、驱动、权重只读，项目写入位于wio。Docker daemon既有镜像/元数据仍由宿主管理。
发布归档仅含约MiB级源码、输入、日志和小型证据，不含权重、镜像或编译缓存。
