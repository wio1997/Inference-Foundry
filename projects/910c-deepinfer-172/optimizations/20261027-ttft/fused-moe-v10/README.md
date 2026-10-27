# fused-moe-v10

状态：ACCEPTED；当前部署：否。

复用固定镜像 dispatch_ffn_combine，替换 eager MoE 的分离 dispatch、动态量化、GMM/激活和 combine。转换一次 FP32 scale 的 INT64 位表示；保持 INT8 NZ 权重。融合覆盖 prefill 和拒绝后的 eager 修复回算，捕获的目标验证图沿用原 MC2。

固定172、16逻辑910C、DP2×TP8/EP16，AISBench32768输入/1024输出、并发8、greedy/ignore_eos、流式、两轮。
每轮实际恢复235904/262144个前缀token（89.990234375%），16/16完整输出与冻结vLLM一致。

| 轮次 | 输出token/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|
| measure1 | 962.19 | 2881.00 | 4.40 | 7369.40 |
| measure2 | 818.54 | 2761.00 | 6.00 | 8926.20 |

合并吞吐 **884.57 token/s**，平均TTFT **2821.00ms**。
合并吞吐按 `16384 / (8.5139362 + 10.0080751)` 计算。
相对上一接受版本，TTFT变化-34.64%，吞吐变化+19.86%。

验收脚本最初漏算 eager 修复回算，随后又将空闲 DP scratch 的既有短上下文路径误作失败。两次均为客户端验收条件问题，未重跑正式测量；两轮原始结果由独立验证器重新逐请求核验，历史错误文件保留在原始归档。融合调用数为 `40 × (prefill次数 + repair次数)`。

源码提交 `0ea97796d0f9e37e8e68cc07c1febf5394bf92d8`，增量补丁基于 `a713eca2b6ddfd5732922859a90237536619bc2e`；DeepInfer上游未推送。
[增量补丁](source/incremental.patch)、[运行命令/配置](config/runtime.json)、[独立核验](results/independent-validation.json)、[完整原始证据SHA](evidence/archives.json)。
控制器保存实际执行版本；新复现必须使用新的控制、日志和结果目录，通过设备归属检查后启动，避免覆盖冻结数据。
权重、镜像和大日志不进入Git；原始完整请求/输出、日志与源码校验表保存在172的`wio/910c-deepinfer-172/final/fused-moe-v10`。

仅Engram表留在CPU，主干和专家常驻NPU，无backbone CPU prefetch。
原固定镜像`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，本地image ID见运行配置；复用编译算子，不启动vLLM引擎。
此为固定重复上下文的两轮结果：Native使用prompt lookup63与64-query目标图验证，vLLM使用DSpark5；不外推到任意prompt或一般模型准确率。
