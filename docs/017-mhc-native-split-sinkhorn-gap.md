# 完整图中的 mHC split/Sinkhorn 平台选择

G334D实际完整verify forward通过RoPE兼容入口，随后mHC mix_stats调用hc_split_sinkhorn。固定镜像默认进入TileLang kernel，其lazy loader明确报tilelang未安装；D实际failed_released且8卡无进程。P API已开放仍不作为完整模型或PD验收。

同一原生mhc模块已有_hc_split_sinkhorn_triton，与TileLang的pre/post sigmoid、初始row-softmax加eps、column normalize及后续19次row/column归一化对应。当前native dispatcher只为gfx1250选择该实现，NPU未路由到它。G346准备仅NPU/FP32/HC4/20iter的复用入口，显式保证输入contiguous并保留零token形状；既不安装CUDA TileLang，也不新写通信/数学kernel。

组件测试计划涵盖0/1/16/96/8192行、普通/大幅度logits、非连续输入、FP64参考、原地改变mix/scale/base后的图回放。0token仅验空形状，不对空的独立kernel创建无操作图。当前候选尚无通过结论，更未验收完整目标/草稿图、原生低比率source/indexer、PD和性能。

- [实际完整图失败](../evidence/full_model_G334_D/server.log)
- [实际清理状态](../evidence/full_model_G334_D/state.json)
- [原生Triton入口复用候选](../code/sg_mhc_compat_g346/mhc_compat.py)
- [组件验证范围](../code/sg_mhc_compat_g346/probe.py)


## G346 组件实际结果

8张910B1全部通过：80用例，64次非空图捕获、192次改变mixes/scale/base回放；16个零token用例只验空形状。普通/大幅度logits、非连续输入均覆盖。相对FP64参考，最大绝对误差9.834766387939453e-7，低于预设rtol3e-5/atol3e-6；controller stopped_released、全部8卡无进程。

G353准备了仅V4.1 NPU backend factory安装的原生port入口，严格核验实际固定mhc源码SHA。其余设备保持原入口，真实128页与RoPE兼容、量化、EP/DSPARK/完整图参数保留。此候选尚未部署：下一次完整装载前先核对native mHC combine/post接口，减少对另外缺失算子的盲目全权重重载。完整模型/PD/质量/性能仍未通过。

- [父级组件验收与数值边界](../evidence/mhc_component_G346/parent_acceptance.json)
- [rank0全部原始用例和变化回放](../evidence/mhc_component_G346/evidence/rank0.json)
- [组件实际释放](../evidence/mhc_component_G346/state.json)
- [准备但未部署的安装范围](../code/sg_pd_g353/mhc_install.py)
- [候选待验收状态](../code/sg_pd_g353/candidate_status.json)
