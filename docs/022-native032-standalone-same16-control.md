# 原生 0.32 单节点同题质量对照

固定 910B1 / aarch64、8 卡、同一 1017 W4A8 权重，在原生 vLLM-Ascend `v0.32.0rc1` arm64 镜像 `d5492ecf` 上启动 P1121 单节点对照。它不是原 P/D 性能 epoch630 的替代结果。[实际启动、113 项源码核对与健康状态](../evidence/native_same16_G1134/G1125_actual_native_CPU_2203_parsed.json)和[完整配置](../code/native_standalone_G1121/candidate.json)可核查；无源码覆盖。

## 实际结果

串行发送既有 8 条算术和 8 条原样抄写提示，`temperature=0`、`seed=0`、`thinking=false`、`max_tokens=128`，SSE 标准接口、无重试。16 条全部正常结束，16 条文本严格等于预设答案，无截断。[父级复算](../evidence/native_same16_G1134/G1134_parent_native_same16.json)逐条从原始 SSE 重建文本、DONE、usage 和 finish reason，并核对 SHA；[完整请求、SSE 和结果索引](../evidence/native_same16_G1134/raw_manifest.json)保留原始证据。首题返回 72，抄写 `K0E411E800` 完整返回。

此配置 TP8/EP8、PP1/DP1/PCP1、NoSP、eager、prefix/async 开启、max-seqs16、max-context263296；max-batched-tokens8192、mem0.88。它只控制选定短题语义，不能作为完整 GSM8K 得分或性能收益。API 未提供缓存详情的字段不能解释为零命中。SGLang 的部署与响应由另一框架分支维护；双方 CANN、torch_npu、索引量化及调度仍存在差异，单次对照不能确定唯一根因。

## CPU 客户端期限纠正

旧 CPU503 的控制器在 01:28UTC 到期退出、容器停止且四核 owner 释放；控制器期限早于容器 sleep 期限。最初按“旧容器仍运行”准备的刷新操作在前置断言处停止，未执行停止或重启。[原始失败](../evidence/native_same16_G1134/G1134_CPU1122_initial_preflight_failure.json)和[实际到期核对](../evidence/native_same16_G1134/G1124_CPU503_EXPIRY_RECONCILIATION_JSON_parsed.json)保留。CPU1122 新租约使用相同镜像、4CPU/32GiB、无 NPU、旧题集只读；[实际预检](../evidence/native_same16_G1134/raw/control/preflight.json)包含容器身份、四核 owner/FD、1319 题 SHA 和就绪时点。正式 AISBench 资产尚未重新验收，不能将诊断就绪写成正式压测就绪。

## 原生压缩 KV 源码参考

新增只读核对 [attention/dsa_v41.py](../code/fixed_native_attention_G1130/attention/dsa_v41.py) 的压缩 source 路径、[compressor](../code/fixed_native_attention_G1130/models/deepseek_v41/compressor.py) 与 [cache_config](../code/fixed_native_attention_G1130/models/deepseek_v41/cache_config.py)：A2 路径归一化后经 RoPE、直接写 BF16 compressed attention cache；packed cache 按硬件能力另行选择。索引量化独立于注意力 KV；不能把 INT8 索引当成 INT8 attention KV。此处是目标固定源码的行为依据，不是任一 SGLang 候选已修复的证明。

原始四档、完整质量及 P8+D8 profile 证据保持原 epoch 和限定范围。SGLang 完整质量、对齐后的四档矩阵、匹配逻辑工作量地图与端到端优化收益仍未完成。
