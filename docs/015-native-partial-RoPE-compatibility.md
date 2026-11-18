# 固定910B依赖栈的 partial RoPE 兼容路径

G314完整D端已越过原生page256断言，进入真实target-verify model forward。首次调用Dsv4NpuRoPE.apply_rotary_mul_inplace时报固定libopapi缺少aclnnInplacePartialRotaryMul；controller随后终态failed_released，全部8卡清理通过。该错误是实际缺失算子，不能通过缩小graph batch或关闭图解决；依赖栈保持固定。

兼容候选复用同一镜像已验证的原生rope_tail，输入仍来自原始BF16 cos/sin缓存，按interleave取成对系数、提升FP32 complex执行旋转，仅一次转回BF16并写回Q/K尾部。没有重新计算角度，也没有在BF16中直接mul/add。非旋转448维与临近内存保持原字节；逆旋转保留原sin符号。原生helper和RoPE类均从实际停止的固定镜像容器取回并核验SHA。

G328在8个物理910B1上验证rows1/16/96/8192×正/逆旋转，共64用例、64图、192次变化输入/系数回放。测试Q使用非连续stride，K共享原地引用；与使用同一BF16缓存系数的FP64 CPU参考比较，实际最大绝对误差为0，非旋转区和邻接canary不变。检查预设rtol0.008、atol0.003；有限测试数据一致不等于所有输入位精确，更没有与不可用custom kernel做bitwise比较。

G334准备完整P/D候选，仅在实际V4.1 NPU backend factory中安装上述RoPE入口，严格核验固定原生源SHA。权重、量化、EP8、DSPARK6、D完整图与PD协议保持原配置。完整model/source/indexer语义、跨机PD、奇数C2状态及DSPARK回滚、质量与四档性能仍未验收；独立RoPE组件不代替这些门槛。

- [G314原始完整模型失败](../evidence/full_model_G314_D/server.log)
- [G314释放状态](../evidence/full_model_G314_D/state.json)
- [8卡组件父级验收及边界](../evidence/RoPE_component_G328/parent_acceptance.json)
- [rank0原始全部用例与变化回放](../evidence/RoPE_component_G328/evidence/rank0.json)
- [组件终态](../evidence/RoPE_component_G328/state.json)
- [固定源码SHA](../code/fixed_RoPE_sources_G331/manifest.json)
- [原生FP32 helper复用](../code/sg_pd_g334/rope_compat.py)
- [只在V4.1 NPU factory安装](../code/sg_pd_g334/integration.py)
