# prefill-sp-v11

状态：ACCEPTED；当前部署：否。

仅在 prefill 按 token 切分 mHC 残差、归一化、路由、共享专家和 Engram 选行/gate；attention 前 all-gather、后 reduce-scatter。KV/索引仍保持完整请求坐标；保持现有验证图。这里尚未实现 vLLM 的 DSA-CP 或 host-UVA Engram。

固定172、16逻辑910C、DP2×TP8/EP16，AISBench32768输入/1024输出、并发8、greedy/ignore_eos、流式、两轮。
每轮实际恢复235904/262144个前缀token（89.990234375%），16/16完整输出与冻结vLLM一致。

| 轮次 | 输出token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|
| measure1 | 1060.74 | 2157.20 | 4.30 | 6587.80 |
| measure2 | 881.89 | 2151.90 | 5.90 | 8207.50 |

合并吞吐 **963.08 token/s**，平均TTFT **2154.55ms**。
合并吞吐按 `16384 / (7.7229358 + 9.2891444)` 计算。
相对上一接受版本，TTFT变化-23.62%，吞吐变化+8.88%。

源码提交 `e2425e5bc5dcdb03dd081c9699eebf53a89e6596`，增量补丁基于 `0ea97796d0f9e37e8e68cc07c1febf5394bf92d8`；DeepInfer上游未推送。
[增量补丁](source/incremental.patch)、[运行命令/配置](config/runtime.json)、[独立核验](results/independent-validation.json)、[完整原始证据SHA](evidence/archives.json)。
控制器保存实际执行版本；新复现必须使用新的控制、日志和结果目录，通过设备归属检查后启动，避免覆盖冻结数据。
权重、镜像和大日志不进入Git；原始完整请求/输出、日志与源码校验表保存在172的`wio/910c-deepinfer-172/final/prefill-sp-v11`。

仅Engram表留在CPU，主干和专家常驻NPU，无backbone CPU prefetch。
原固定镜像`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，本地image ID见运行配置；复用编译算子，不启动vLLM引擎。
此为固定重复上下文的两轮结果：Native使用prompt lookup63与64-query目标图验证，vLLM使用DSpark5；不外推到任意prompt或一般模型准确率。
