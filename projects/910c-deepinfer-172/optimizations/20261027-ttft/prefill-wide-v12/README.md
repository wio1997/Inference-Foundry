# prefill-wide-v12

状态：ACCEPTED；当前部署：是。

在已接受的融合 MoE 和序列并行上，将 chunk1024→4096、MC2 专用窗口5120→1024MiB；按完整路由上界把融合接收容量65536→196608。固定负载的 prefill 从4次完整40层前向减为1次。

固定172、16逻辑910C、DP2×TP8/EP16，AISBench32768输入/1024输出、并发8、greedy/ignore_eos、流式、两轮。
每轮实际恢复235904/262144个前缀token（89.990234375%），16/16完整输出与冻结vLLM一致。

| 轮次 | 输出token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|
| measure1 | 1078.66 | 1921.40 | 4.40 | 6405.90 |
| measure2 | 928.39 | 1755.30 | 5.90 | 7798.60 |

合并吞吐 **997.90 token/s**，平均TTFT **1838.35ms**。
合并吞吐按 `16384 / (7.5946399 + 8.8238842)` 计算。
相对上一接受版本，TTFT变化-14.68%，吞吐变化+3.62%。

源码提交 `807448d4073ba461388791f3a1d372800ba71da4`，增量补丁基于 `e2425e5bc5dcdb03dd081c9699eebf53a89e6596`；DeepInfer上游未推送。
[增量补丁](source/incremental.patch)、[运行命令/配置](config/runtime.json)、[独立核验](results/independent-validation.json)、[完整原始证据SHA](evidence/archives.json)。
控制器保存实际执行版本；新复现必须使用新的控制、日志和结果目录，通过设备归属检查后启动，避免覆盖冻结数据。
权重、镜像和大日志不进入Git；原始完整请求/输出、日志与源码校验表保存在172的`wio/910c-deepinfer-172/final/prefill-wide-v12`。

仅Engram表留在CPU，主干和专家常驻NPU，无backbone CPU prefetch。
原固定镜像`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，本地image ID见运行配置；复用编译算子，不启动vLLM引擎。
此为固定重复上下文的两轮结果：Native使用prompt lookup63与64-query目标图验证，vLLM使用DSpark5；不外推到任意prompt或一般模型准确率。
