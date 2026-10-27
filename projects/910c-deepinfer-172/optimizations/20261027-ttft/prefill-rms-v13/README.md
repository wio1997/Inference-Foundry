# prefill-rms-v13

状态：INCONCLUSIVE；当前部署：否。

对兼容的 NPU BF16 输入和同 dtype gamma，使用固定镜像 npu_rms_norm_cast，保留 BF16 结果与 FP32 中间结果。混合 dtype 权重保留原数学路径，不将高精度 gamma 强制转为 BF16。覆盖 prefill 与目标验证中的 RMSNorm。

固定172、16逻辑910C、DP2×TP8/EP16，AISBench32768输入/1024输出、并发8、greedy/ignore_eos、流式、两轮。
每轮实际恢复235904/262144个前缀token（89.990234375%），16/16完整输出与冻结vLLM一致。

| 轮次 | 输出token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|
| measure1 | 1137.26 | 1958.70 | 4.10 | 6144.30 |
| measure2 | 961.82 | 1778.90 | 5.60 | 7484.80 |

合并吞吐 **1042.21 token/s**，平均TTFT **1868.80ms**。
合并吞吐按 `16384 / (7.203249 + 8.517179599999999)` 计算。
相对上一接受版本，TTFT变化+1.66%，吞吐变化+4.44%。

本候选未选为当前部署：没有观察到首token收益；约30ms差异不足以在两轮中判定稳定差异。输出吞吐虽更高，按本次TTFT目标保留v12。

源码提交 `715a039916b915592f5257e662728d7826913a20`，增量补丁基于 `807448d4073ba461388791f3a1d372800ba71da4`；DeepInfer上游未推送。
[增量补丁](source/incremental.patch)、[运行命令/配置](config/runtime.json)、[独立核验](results/independent-validation.json)、[完整原始证据SHA](evidence/archives.json)。
控制器保存实际执行版本；新复现必须使用新的控制、日志和结果目录，通过设备归属检查后启动，避免覆盖冻结数据。
权重、镜像和大日志不进入Git；原始完整请求/输出、日志与源码校验表保存在172的`wio/910c-deepinfer-172/final/prefill-rms-v13`。

仅Engram表留在CPU，主干和专家常驻NPU，无backbone CPU prefetch。
原固定镜像`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，本地image ID见运行配置；复用编译算子，不启动vLLM引擎。
此为固定重复上下文的两轮结果：Native使用prompt lookup63与64-query目标图验证，vLLM使用DSpark5；不外推到任意prompt或一般模型准确率。
