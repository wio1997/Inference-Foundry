# Flash 双线执行边界与性能准备

A 是 14P/15D 的 vLLM 性能线，B 是 46P/47D 的 SGLang 适配线。A 不等待 B 的压缩故障、reorder、Graph 或 80K。两线只在重型 NPU/SDMA 与正式测量窗口分时，后台实际负载如实记录，不操作无关任务。

R080 的 eager、逐 token KDA、CPU tensor hash/trace 是数值参考。A 候选采用固定 rc1 官方 P DP2/TP8、D DP16/TP1，D 目标模型 FULL_DECODE_ONLY、MTP D5（draft eager）、原生混合 align/whole-segment 缓存；必要数学修正保留来源和数值门禁。候选未部署资格化。新的 PD SDMA shim 不替换 Store 的原生 Mooncake engine，CPU14项只是边界检查。

A 首次新实际门禁 R125 已完成：内置 QK norm chunk 的输出通过，但 F32 状态未过 1e-4，2/65/2379 token 最大误差约 5.66e-4～7.03e-4；容器 exit0、P14恢复8卡空闲。R126 测试完整 FP32 原生 chunk contract 是否支持及正确，待独立审核一次执行；不会放宽门槛，也不会把 Python 逐 token 参考当成性能路径。

B 的 R122 整模型在第一长 decode 的压缩核报507035，tail store/新score尚未发出。R123 用同一实际 D47/device0 快照只回放一个原压缩核，对CPU64误差0通过。该局部结果不解决整模型的物理格式、alias、stream、SDK注册或生命周期上下文。

正式性能仍为0，实际80000输入/600输出尚未发。必须先通过A自己的PD/Graph/MTP/容量/缓存门禁，再做C16/32/64各4轮新后缀。每档C个独立74400逻辑前缀真实预热，预热与正式分开，自然HBM/DRAM/L3驻留，真实命中或null，不能套用别的模型输入或命中常数。
