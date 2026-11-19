# 同题对照确认语义差异，logprob 开关未消除故障

固定 910B1/aarch64、同一 1017 W4A8 权重；SGLang 镜像 `7da96386`，D1098 仍为 standalone / NoSP / eager、prefix 与 overlap 开启、TP8/EP8。串行执行原有 8 条算术、8 条原样抄写提示，T0/seed0/thinking=false/max128、无重试。[SGLang 原始16条](../evidence/same16_and_logprob_G1137/G1129_SG_same16_raw_2204_parsed.json)全部传输完成，仍仅8条严格匹配：算术3/8、抄写5/8。首题期望72、返回78；`K0E411E800` 返回 `K0E411E`，finish为stop。

固定原生 vLLM 0.32 在 P1121 的[同题对照](https://github.com/wio1997/Inference-Foundry/blob/167121b519e684485e7c9a6cba4b75f794342a88/docs/022-native032-standalone-same16-control.md)为16/16匹配。[父级比较](../evidence/same16_and_logprob_G1137/parent_pair_assessment.json)核对两边16份请求文件逐字节一致、prompt-token计数一致；完整原生响应、启动和源码归原生分支。两组均 NoSP/eager、C1、无传输失败；由此可确认选定负载上的实际框架路径差异，但不能由此确定唯一算子根因，也不能代替完整1319题质量。CANN9.1/post6与9.2/post7、FP4与INT8索引、chunk1024/8192和mem0.80/0.88等差异仍需披露。

## logprob 的受控 A/B

两条原始提示分别执行两次重复，顺序AB后BA，仅改变 `/generate` 的 `return_logprob` 布尔值；其余body逐项相同，`top_logprobs_num=0` 两边一致。8条HTTP完成，cached_tokens均0，观测器未开启。[父级逐body/SHA核查](../evidence/same16_and_logprob_G1137/G1137_parent_logprob_AB_assessment.json)与[原始请求和响应索引](../evidence/same16_and_logprob_G1137/G1116_logprob_AB_raw_2204_parsed.json)保留。

| 提示 | 第一次off | 第一次on | 第二次on | 第二次off |
| --- | --- | --- | --- | --- |
| math0，期望72 | 65 | 78 | 72 | 78 |
| bind1，期望K0E411E800 | K0E411E | K0E411E | K0E411E | K0E411E |

这不支持“开启logprob即可稳定修复”的判断；同一开关内算术输出也变化。抄写四次均提前EOS1。冻结manifest的scope文案沿用了旧fourcase1092名称，实际8条state/原始文件为准；旧原始记录不改写。

## 已有快照的 BOS 检查

仅在CPU客户端复用先前240份真实attention快照，不重发NPU请求。比较两条提示的相同BOS0/位置0、8rank与3个已选层，共24行；两次prefill实际padding分别32/40行。[父级范围说明](../evidence/same16_and_logprob_G1137/G1137_parent_BOS_assessment.json)及[逐行原始分析](../evidence/same16_and_logprob_G1137/G1119_real_BOS_alllayers_2204_parsed.json)可查。第0层Q全部32768元素数值一致；BOS KV只有1/512元素相差0.0001220703125。第2/20层Q最大relative-L2分别约0.01220/0.01334，KV约0.00842/0.00474。不同prompt/padding下的量化批形状差异尚不能解释为唯一故障或首层输入污染。

## 后续定位边界

[实际接口/配置核查](../evidence/same16_and_logprob_G1137/G1117_runtime_head_sources_2204_parsed.json)确认当前未开启hidden-state返回，未发现运行时server-args更新接口；没有发送已知不被支持的hidden-state请求。此前selectedattention/KV-history核对范围保持不变，不外推其他层、LM head或全模型正确。

新线索是压缩attention KV：当前SG写BF16缓存前仍执行per16 E4M3scale/FP4E2M1舍入，而固定原生A2压缩source经RoPE后直接写BF16。索引量化与attention KV是不同路径，W4A8权重也不是KV量化。隔离候选应仅改这一处KV舍入，保留FP4索引/权重/其余算子；候选启动、实际分派和同题结果另行留证，不能仅凭源码宣布修复。

完整质量、最终DSPARK/PD/图对齐、SG四档首次cold/replay、双方匹配profile地图与端到端收益仍未完成。
