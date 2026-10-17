# 官方 PD 已有 trace 的性能线索（2026-10-17）

复用已归档的官方 PD 实验，只读分析，没有新增设备实验。此处是候选选择依据，不是正式性能基线或已实现收益。

来源：`glm53-execution-evidence-backup/20261017-official-pd/node32-official-pd-evidence-v1.tar.gz`（SHA256 `4fd60f8abf5faf882abb8c474009425f5d3f84cdd9b77282b7c92ac02584c715`）及 node33 对应归档（`bb0a10fb1ac4d9b01abb23436f60744f83932c834b179ed4501f259b3b1e36ef`）中的 rank 0 `trace_view.json`。采样方式见 [official_pd_profile.py](../../tools/official_pd_profile.py)。

## 观察

配置为 P/D 各 TP16、DP1，P eager、D FULL_DECODE_ONLY，原版 MooncakeConnectorV2/代理，无 MTP/EP/MC2/FlashComm/DSA CP。模型和缓存维持 W8A8C8。

| 范围 | 观察 | 解释边界 |
| --- | --- | --- |
| D graph | 16次 aclmdlRIExecuteAsync；除首调用外 host 调用约28.6–57.6μs | 首次26.84ms不能混入稳态 |
| D连续decode周期 | 按78个MLA prolog分段，剔除首个混合窗口和尾stop窗口后15个周期45.13–47.54ms，中位约45.87ms | profile内周期，不能当无profiler正式TPOT |
| 周期内Computing | 32.76–33.65ms | CANN派生分类，非完整关键路径证明 |
| 周期内未重叠Communication | 10.10–12.14ms，约22%–26% | 不是全部可消除的开销 |
| 周期内Free | 1.91–2.04ms | 不含被排除的大间隔 |
| TP allreduce | 每完整forward约157次；shared/routed CPU路径各75次 | 可提出MoE分支与重复collective假设 |
| 两类MoE grouped matmul | 各75次，累计约5.36+3.78ms/周期 | 算子累计仅供排序 |
| 路由Cumsum | 75次，累计约2.80–3.57ms/周期 | 未证明是可优化关键路径 |
| KV batchGet | 设备区间403.212μs，host API504.95μs | 单次，位于首replay之前，不是整个P→D等待 |

allreduceAicpuKernel 生命周期累计约41.5–44.5ms/周期，与计算重叠，**不能当成43ms通信关键路径，也不能再与matmul耗时相加**。

热连接1024输入日志为每rank55,836,672B、1.80–4.63ms；首次256输入READ为1.245–1.306s并伴随建连。正式基线应把冷连接首请求单列，连接预热之后仍可使用冷前缀；两种“冷”不能混淆。

采样含native arming：P有两个完整forward，D有一个eager forward、16次replay及244.74ms的npu_fx_compiler inference，整窗不能作为纯PD基线。耗时仅审查rank0，未证明最慢rank；主机时钟偏差不允许直接跨机相减，当前没有C4性能结论。

## 下一决定

执行线程先复用恢复后的服务完成无profiler的256/64、1024/64、C1/C4基线。现有证据优先指向D端MoE/TP通信与分支组织，不能据冷建连耗时先优化SDMA载荷带宽。

候选之一是已有 `multistream_overlap_shared_expert`，先确认在TP16、无EP、GLM W8A8C8实际路径是否生效；它与fused MC2互斥。不适用就直接排除，不跑仅修改开关的无效对照。随后可核已有 `enable_shared_expert_dp` 是否减少shared分支collective，或依据MTP支持/接受率选择减少target执行次数的候选。一次有解释的机制对照优先，不批量打开无关功能。

有效收益需要同条件TTFT/TPOT/吞吐及正确性证据，同时验证目标分支重叠、collective数或target执行次数实际改变。部署成功、trace解释和PR归档都是阶段进度，不表示原性能优化目标完成。
