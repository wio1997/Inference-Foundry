# GLM-OPT-0002：精确20轮Sinkhorn融合

## 原理介绍

原实现对每次mHC pre分别提交行softmax、epsilon、39次归约与除法。真实8步中720次pre对应28,080次除法。候选把这一小矩阵循环放入一个Triton Ascend核，减少提交和独立任务；是否约束最终关键路径仍需端到端数据。

## 源码变更

只替换Sinkhorn循环，保留20次迭代、epsilon位置和FP32运算；投影、pre/post系数和残差混合不变。[补丁](../../../patches/exact-sinkhorn.patch)、[候选](../../../code/sinkhorn/mhc_candidate.py)、[融合核](../../../code/sinkhorn/fused_sinkhorn.py)、[基线](../../../code/sinkhorn/mhc_baseline.py) 均保留。原生`npu_mhc_pre`输出未做Sinkhorn，不能直接替换。

## 实测优化范围

实际910B固定A2镜像通过27个数值、4组图回放、5组完整适配器用例；系数最大误差约1.49e-7。随后完整模型通过数学题、127/129/131/8191/8193边界及32768输入/32输出的7个P/D用例。微计时含主机提交间隙，未当作端到端收益。

## 波动噪声

新模型前三轮预热仍有时长和TPOT漂移，未达到计分门槛；没有最终同臂噪声标定。原始预热不删除、不挑选最后一轮当正式结果。

## 结论与适用边界

裁决为INCONCLUSIVE。组件与所列P/D检查成立，整机收益、完整四档对齐和广泛精度仍未验证。固定CANN9.1与vLLM的CANN9.2不能互换二进制。

## 代码与链接

[组件证据](../../../evidence/sinkhorn_component_results.json)、[P/D证据](../../../evidence/model_PD_acceptance.json)、[预热门槛失败](../../../evidence/failed_warmup_stability.json)。Run索引见 [records/runs.json](../../runs.json)。代码来自实际测量文件的相同SHA副本；完整先前适配运行时尚未在此增量分支重建验收。后续为补足预热后首轮冷测/热重放，再检查融合命中与其余三档。

补充诊断：30种逻辑MTP长度的编译预热未覆盖实际分配长度256/512。G37的两轮时长129.6500/124.2774秒，原5%三指标门槛通过，但第八轮仍编译一个分配核，因此没有提交正式数据。以原生缓存hash恢复实际特化参数，记录见[键恢复证据](../../../evidence/allocator_key_recovery_g39.json)，不将“编译脚本完成”当作服务充分预热。
