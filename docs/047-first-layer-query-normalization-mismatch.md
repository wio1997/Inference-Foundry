# 第0层 Query 归一化不一致：真实逐层证据与条件修正

固定910B1/aarch64、同一1017 W4A8权重与既定两个镜像；以standalone/NoSP/eager的P1146原生和D1147 SGLang作数值对照，不把该配置当最终DSPARK/PD性能配置。[原生实际数据](https://github.com/wio1997/Inference-Foundry/blob/2ad3cac312dd0fce028852f9f0910cd1e0b22273/docs/023-native032-layerwise-observation.md)由其独立分支维护；本分支维护SG响应、跨框架CPU分析与候选。当前发现的是具体架构契约不一致，条件修正已在同一组16条短题通过；完整质量尚待验证。

## 相同前缀与真实边界

选取既有math1“14×6+7”的精确30个token、位置0至29，max1、T0/seed0、串行无重试。未开启观察时原生返回91、SG返回96；打开逐层观察后原生仍91、SG返回92；随后打开已有backend操作数观察时SG返回98。SG三次cached_tokens均0。观察方式和阶段同时变化，不能由单组响应变化证明同步是唯一原因，也不能把max1首预测当全量质量得分。[未观察SG响应](../evidence/query_norm_mechanism_G1167/unarmed/SG_response.json)与[观察SG响应](../evidence/query_norm_mechanism_G1167/armed/SG_response.json)保留。

双方各40层×8rank，共各320份真实attention归一化输入/投影输出，均取真实30行而非padding尾行。每端内部，各层8rank的输入/输出原始BF16字节SHA一致，允许在这一明确字节链内用rank0做CPU比较。[SG全部rank索引](../evidence/query_norm_mechanism_G1167/G1150_layer_snapshots_2204_parsed.json)与[CPU逐层结果](../evidence/query_norm_mechanism_G1167/G1153_CPU_layer_comparison_result_2204_parsed.json)可核查。第0层全部153600个输入元素相同，但输出relative-L2为0.47049859；位置0为0.18048337，最后真实行29为0.49576473。后续层的输入已经不同，不能将后续偏差直接归因于其本层算子。

全部向量保留受控服务器，公开库只记录SHA、元数据、源代码和复算结果。复制会同步NPU，采样延迟不进入性能比较。当前图均关闭，不是图回放证明。

## 第0层位置0的独立分解

该层ratio0，无压缩KV/索引，位置0的RoPE为恒等，只允许当前1个KV。CPU客户端读取同一检查点参数；Q-b为W8A8_DYNAMIC，实际weight_offset全0，其余相关投影/归一化/sink为FLOAT BF16。独立实现保留BF16阶段、对动态INT8投影作精确整数CPU累积，并单独披露最终TP归约及FP32/GEMM舍入差别。[参数契约](../evidence/query_norm_mechanism_G1167/G1154_layer0_checkpoint_contract_2204_parsed.json)和[独立参考](../evidence/query_norm_mechanism_G1167/G1156_layer0_BOS_CPU_result_2204_parsed.json)保留，未以相同错误实现换dtype当oracle。

| 位置0的检查 | 实际结果 |
| --- | --- |
| 原生输出与checkpoint独立参考 | relative-L2 0.00441239 |
| SG输出与同一参考 | relative-L2 0.20201126 |
| 八rank真实Q与参考Q | relative-L2 0.28535至0.44359 |
| 真实KV与参考KV | relative-L2约0.0000216744 |
| 实际sink与checkpoint对应head | 全部相同 |
| attention核在真实Q/K/sink上的独立计算 | 全部64个真实head的BF16结果数值一致 |
| 用真实attention输出独立复算WO-A/B与SG最终输出 | relative-L2 0.00383019 |

这些检查只覆盖该真实BOS位置；不宣称所有算子、所有位置或完整模型正确。[八rank逐阶段原始结果](../evidence/query_norm_mechanism_G1167/G1160_BOS_components_result_2204_parsed.json)支持主要偏差位于Query路径，而非此处的KV、sink或给定操作数上的attention核。WO-A/B的复算只作数值接近度，不将FP32/BF16归约舍入解释为bitwise要求。

## 具体配置不一致

SG的[V4.1配置](../code/sglang/srt/configs/deepseek_v41.py)默认`q_head_norm=False`；其通用`_compute_q_b`也按该标志决定是否做每头RMSNorm。固定原生0.32的project-Q在Q-LoRA归一化和Q-b投影后直接RoPE，不追加每头归一化。然而当前固定SG源码两个NPU方法`_forward_prepare`和`_forward_prepare_multi_stream_npu`无条件执行`npu_rms_norm(q,ones,eps)`，绕过了这一标志。

独立Q的各头RMS范围为0.95974至2.46375，SG实际为0.99848至1.00075；把独立参考按head做相同RMSNorm后，与实际Q的relative-L2降为0.00161750。[实际CPU验证](../evidence/query_norm_mechanism_G1167/G1161_Q_head_norm_CPU_2204_parsed.json)与[父级机制核查](../evidence/query_norm_mechanism_G1167/G1162_parent_query_normalization_mechanism.json)支持这是一处具体配置契约不一致。旧对象标志未通过对象接口直接导出，默认配置及无覆盖参数为依据；新候选将记录真实配置和执行分支。

## 最小修正与待验收范围

[核心两处条件patch](../code/Q_head_norm_policy_G1163/deepseek_v4_q_head_norm.patch)仅在`self.q_head_norm`为True时执行每头RMSNorm；Q-LoRA norm、KV norm、RoPE、权重、索引与压缩KV精度保持。实际兼容层[安装实现](../code/Q_head_norm_policy_G1163/q_head_norm_policy_g1162.py)固定源码SHA、断言每个方法只改1处、保持函数对象身份，记录实际分支，不改镜像源码。True配置的数学行为保持原状。

[D1147完整源码索引](../evidence/query_norm_mechanism_G1167/D1147_source_index_G1167.json)与[D1163候选索引](../evidence/query_norm_mechanism_G1167/D1163_source_index_G1167.json)保留原始/公开SHA，复用未变源码。候选实际已就绪、8rank各1384参数严格装载且安装记录存在。[当前状态](../metadata/Q_head_norm_policy_G1167.json)记录实际同题16/16与八rank False/skip；完整修复验收仍需全1319质量与最终DSPARK/PD/图配置。当前只改target两个NPU分支，draft路径和最终图回放需另行核对。

未完成项仍包括：完整质量、最终优化对齐、SG四档独立充分预热后的首次cold/replay、匹配P8+D8逻辑工作量profile地图与端到端收益。

## 条件修正后的同题16条实际验证

D1163在关闭观察、NoSP/eager/standalone、C1串行、T0/seed0、max128下复跑原16条：8条算术和8条记录绑定全部精确匹配预期，16次均有SSE DONE与stop，未发生传输失败。原始请求JSON字节与D1136控制组16条全部相同；旧三方字节核查见既有控制组档案。旧D1098为8/16，单独BF16压缩KV D1136为7/16，当前D1163为16/16；原生固定对照为16/16。该结果只证明所列短题，不能外推全量质量或端到端收益。

[父级逐条SSE重建](../evidence/Q_head_norm_same16_G1170/G1170_parent_Q_policy_same16_audit.json)核对实际内容、usage、finish与SHA；[完整原始证据及八rank分支](../evidence/Q_head_norm_same16_G1170/G1168_Q_policy_same16_raw_2204_parsed.json)显示rank0至7对象的`configured_q_head_norm=false`且实际执行`skip_per_head_RMSNorm`，helper源SHA固定。单流路径实际触发，多流路径仅完成安装检查，未运行；最终DSPARK、图及PD尚待验证。

## 修正后的真实Query与独立参考

用完全相同30-token输入、BOS位置0重新采集24个backend操作数和40层×8rank边界，[采集记录](../evidence/Q_head_norm_same16_G1170/G1172_QKV_capture_2204_parsed.json)确认实际退出、双观察器关闭，首预测为正确91。复用同一checkpoint CPU参考和同一计算源码，只更改输入证据路径。[实际复算](../evidence/Q_head_norm_same16_G1170/G1173_postfix_BOS_CPU_result_2204_parsed.json)显示八rank Query relative-L2为0.00118382至0.00143659，原条件修正前为0.28535至0.44359；KV仍约0.0000216744，sink相同，给定实际Q/K的核结果仍数值一致。修正后SG第0层BOS输出与同一参考relative-L2为0.00435270，原生为0.00441239，旧SG为0.20201126。

这里的变化支持已定位机制，参考自身包含明确BF16/TP舍入边界；未设置事后全模型容差，未将接近度升级为完整质量或性能通过。之后的C16、完整1319和最终DSPARK/PD/图仍需独立验收。
