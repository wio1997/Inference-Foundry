# BF16 压缩 KV 候选：路径生效，语义问题未修复

在同一 SGLang Ascend daily A2 镜像、910B1/aarch64、1017 W4A8 权重上，仅移除压缩 attention KV 的额外 FP4 舍入，保留 FP4 索引、BF16 物理缓存、compressor/RoPE、W4A8 权重及其余算子。[基线同题对照](045-same16-native-control-and-logprob-AB.md)是原生16/16、SG8/16。本候选实际为7/16，不能认作质量修复或性能优化。

## 动机与改动范围

[固定原生 A2 源码](https://github.com/wio1997/Inference-Foundry/blob/167121b519e684485e7c9a6cba4b75f794342a88/code/fixed_native_attention_G1130/attention/dsa_v41.py)的压缩 source 经归一化、RoPE 后写 BF16 attention cache；原SG的 `_rope_fq4(...compressed_kv=True)` 在写BF16缓存前仍做per16 E4M3scale/FP4E2M1舍入。该差异提供了一个可推翻的假设：去掉额外舍入可能消除选定语义故障。attention KV精度、索引量化、权重量化是不同契约；本候选仅改第一项，不声称两框架所有量化行为已对齐。

[完整60文件源码索引](../evidence/BF16_compressed_KV_G1142/G1142_complete_candidate_source_index.json)复用51份未变文件，新增9份epoch/候选文件。实际数学改动在[npu_backend.py](../code/BF16_compressed_KV_G1136/npu_backend.py)：NPU compressed-KV quant函数对post-RoPE BF16值作identity返回，其他设备保留原函数；FP4 index代码、池布局与W4A8 checkpoint不变。CLI argv与D1098一致；NoSP、standalone、TP8/EP8、eager、prefix/overlap仍开启，chunk1024、mem0.80不变。候选JSON中PREPARED字段是部署前快照，实际状态以下列runtime证据为准。

## 首次启动失败与源码校验修正

D1098全部诊断已结束、原始文件冻结且两端idle后受控退休，八卡owner释放，原容器/cache保留。首次D1132启动在模型初始化前被可选观测器的旧backend SHA断言拦截，随后自动停止并释放八卡；没有target装载或质量请求。[实际失败与完整异常](../evidence/BF16_compressed_KV_G1142/G1136_old1132_terminal_parsed.json)保留，不能将其算成KV假设失败。

D1136只更新观测器期望SHA以匹配精确候选文件，保留原断言，观测器函数AST不变；旧失败目录不覆盖。[修正记录](../evidence/BF16_compressed_KV_G1142/G1136_source_pin_correction.json)和[实际就绪](../evidence/BF16_compressed_KV_G1142/G1138_BF16_D_observation3_2204_parsed.json)记录8rank各1384参数严格装载、missing为空，FP32 C2 ring8/请求槽17/固定1794048B保持。启动期`freeze_gc`有一次连接尚未监听的日志，后续/ready与真实16请求正常；不将它误报为终止。

## 实际调用与原始结果

实际八卡均生成一次compressed-KV调用记录，输入为BF16、末维512，helper SHA与部署文件一致。记录证明新路径被调用；不是仅靠import或配置推断。[原始冻结索引及八卡调用记录](../evidence/BF16_compressed_KV_G1142/G1141_BF16_SG_same16_raw_2204_parsed.json)与[父级逐请求比较](../evidence/BF16_compressed_KV_G1142/G1142_parent_BF16_candidate_assessment.json)保留；请求body与基线SG、固定原生16份逐字节一致，标准SSE，C1/max128/T0/seed0/thinking=false、无重试。

| 项目 | 基线SG D1098 | BF16候选 D1136 | 固定原生P1121 |
| --- | --- | --- | --- |
| 正常完成 | 16/16 | 16/16 | 16/16 |
| 按原严格文本标准匹配 | 8/16 | 7/16 | 16/16 |
| math0，期望72 | 78 | 72 | 72 |
| bind1，期望K0E411E800 | K0E411E | K0E411E | K0E411E800 |

候选math6返回`19*11=209; 209+7=216.`，计算正确但违反原定“只返回整数”的严格匹配；未临时改评分。其余算术错误与多处抄写尾部遗漏仍在。单次16条不支持统计质量退化结论；它足以否定“该单一改动已修复语义问题”。没有据此重跑完整1319题或提交正式AISBench。

## 当前状态与后续

[当前候选状态](../metadata/BF16_compressed_KV_G1142.json)将该路线标为路径生效、未成为质量修复；D1136仅保留诊断，不能当有效性能baseline。全部请求结束，nativeP1121、CPU1122和canonical guard保留；guard旧D1098标签是历史身份，实际资源以每阶段PID/CID/owner与FD证据为准，不能据旧标签重启或抢占。

后续使用相同teacher-forced前缀与位置比较两框架实际逐层输入/输出，找首个分歧；先核实实际分派与同步，复用已有operand observer。自由生成分叉后的不同历史不用于推断算子因果。恢复最终DSPARK/PD/图执行、完整1319质量、SG四档首次矩阵、匹配逻辑profile地图和端到端收益仍未完成。

末态[实际D/CPU核对](../evidence/BF16_compressed_KV_G1142/G1143_final_D_runtime.json)确认控制器和容器存活、请求及队列为0、CPU客户端只剩sleep、观测器未开启。只读源码显示`SGLANG_NPU_USE_MULTI_STREAM`默认False，八个实际worker环境未覆盖该变量；实际函数分派尚未采样，后续逐层对照一并记录。
