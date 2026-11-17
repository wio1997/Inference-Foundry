# 原生权重覆盖、910B KV 字节预算与 C1/C2 接入边界

固定条件：A2 daily 镜像、910B1、TP8/EP8、BF16 KV、context 263296、C16、DSPARK 5 predictions（SG 参数 6 含 anchor），D target/draft full graph。没有改变检查点压缩比、量化或输出要求。

## 真实完整权重加载通过，服务尚未通过

G82 两端各 8 个 target 与 8 个 draft 覆盖凭证均 passed，分别每 rank 1384 与 73 个参数，missing 为空。目标权重每卡约 42.32 GiB，草稿约 2.82 GiB。随后原生 KV 分配 OOM，两个 controller 最终 failed_released，各自全部 8 卡进程清空。权重覆盖不证明生成、PD、图或性能正确。

- [32 份覆盖汇总](../evidence/full_model_G82/all32_load_coverage_summary.json)
- [P 原始日志](../evidence/full_model_G82/2203/server.log)
- [D 清理结果](../evidence/full_model_G82/2204/state.json)

## 可复现的容量估算差异

原生通用规划器按 FlashMLA row 448 + 64×2 + 8 = 584 bytes 计价；910B 的 PA_ND BF16 实际 row 为 (448+64)×2 = 1024 bytes。低压缩比 indexer 的基类 packed buffer 为 68 bytes，NPU 实现还保留 INT8 K 128 bytes 和 FP16 scale 2 bytes，源码派生合计 198 bytes；该 indexer 在 G82 OOM 前尚未构造，不把源码推导称为现场分配观测。

owner copies 为 C2 的三个 1/2 加 C1 的一个 1，共 2.5。保留原生 41/40 系数后，旧计价 (584+68)×2.5×41/40 = 1670.75 bytes/full-token，修正源码估算为 (1024+198)×2.5×41/40 = 3131.375。独立整数形状推导出的单个 C2 tensor 为 P 3088711680 bytes、D 3148349440 bytes，精确对应日志中的 2.88/2.93 GiB OOM 申请。

- [原生规划器源码](../code/native_KV/G86_native_planner_sources/sglang/srt/model_executor/pool_configurator.py)
- [实际原生三段方法和独立分配算术复现](../scripts/reproduce_native_KV_pricing.py)：在仓库根运行 `python3 scripts/reproduce_native_KV_pricing.py`，仅 CPU，不构造模型或申请 NPU。
- [原始复现输出](../evidence/KV_accounting_G94/native_pricing_replay.json)
- [限定架构/模型/BF16 的候选修正](../code/KV_accounting_candidate_G94/kv_geometry_g94.py)：尚未部署到完整模型；CPU 检查仅证明安装与函数绑定，未验证实际容量或 OOM 已解决。

## 仍需实现的模型契约

检查点需要 C1/C2，固定 Ascend backend 的 forward 仅允许 0/4/128，且没有模型要求的 forward_low_ratio_sources。NPU pool 的 low-ratio factory 参数、compressed buffer accessor、PD contiguous metadata 也需按实际 C1/C2 布局接入。C2 的 PD wire page 是全局 128 token / 2 = 64 rows，不能直接把物理 PA_ND 128-row page 当成一份 wire item。禁止仅删除 guard 或修改检查点压缩比绕过。

- [固定 attention backend](../code/native_KV/G88_native_attention_sources/sglang/srt/hardware_backend/npu/attention/attention/ascend_dsv4_backend.py)
- [固定 NPU KV pool](../code/native_KV/G85_native_KV_sources/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)

## 探针与数据准备范围

G89 在 custom API 未注册处退出，没有执行压缩比 case，不能据此宣称硬件不支持 C1/C2。G92 使用固定镜像原生 DSpark builtin API，C128 baseline 执行时报告 sinks is required，仍没有 C1/C2 执行证据。G96 加入原生 init_npu_backend 初始化后，全部8个 rank 的 custom target API 确实注册成功；加入 FP32 -inf head sinks 后，builtin DSpark baseline 明确要求64个query heads，拒绝探针原先未 padding 的8-head输入。原生模型 _kernel_num_heads 与 _local_attn_sink 在 TP8 下已经把每rank8heads pad到64，因此该失败仍是探针输入契约问题，不是压缩比不支持的证据。两代失败均已退出并清空全部8卡。G100 按实际模型的64-head契约复验；未见结果前不声称 baseline 或 C1/C2 通过。constant-value fixture 只能检查弱数值不变量和 changed-input graph，不验证 causal mask 或完整 attention 语义。

- [G92 原始 rank0 失败](../evidence/SFA_operator_G92/evidence/rank0.json)
- [G92 最终清理](../evidence/SFA_operator_G92/state.json)
- [修正探针](../code/SFA_operator_G96/probe.py)

独立 CPU 客户端在 228.575 秒内准备完 4 档、24 份 profile，448 个请求首 128 token 前缀全局唯一，两个框架复用同一字节。尚未发送模型请求，服务端实际 usage 未验证。

- [数据 manifest](../evidence/CPU_client_G83/dataset_manifest.json)
- [全局前缀检查](../evidence/CPU_client_G83/cold_prefix_audit.json)
- [CPU 任务完成状态](../evidence/CPU_client_G83/completion_state.json)

- [G96 原生 API 注册及 head 契约失败](../evidence/SFA_operator_G96/evidence/rank0.json)
- [G96 最终清理](../evidence/SFA_operator_G96/state.json)
- [原生模型 head padding 与 sink 实现](../code/native_KV/native_model/deepseek_v4.py)
- [G100 与模型一致的探针](../code/SFA_operator_G100/probe.py)

## G104 实际固定二进制契约矩阵

在8张独立物理910B1上，按原生模型64-head padding、FP32 sinks、PA_ND BF16、C1/C2/C4 top-k512索引契约分别执行 target custom device metadata 与 draft builtin host metadata。target四种ratio均因libopapi缺少aclnnSparseAttnSharedkvMetadata符号失败。builtin的C128和C4各在全部8卡上通过常量数值、1次capture和3次changed-input replay，合计16个capture/48个replay；C1/C2均在metadata scheduling报invalid params。controller正常结束并全部8卡进程清空。这里的constant fixture不证明mask或任意Q/KV数值正确，注册成功也不等于相应CANN符号存在。

- [G104 rank0 完整8种契约结果](../evidence/SFA_operator_G104/evidence/rank0.json)
- [G104 终态与清理](../evidence/SFA_operator_G104/state.json)
- [正确top-k与两条原生API的探针](../code/SFA_operator_G104/probe.py)

下一步验证固定原生算子能否分别计算SWA与显式索引compressed partition并使用返回的log-sum-exp合并，保留同一检查点的C1/C2 latent与合法候选。任何坐标映射都必须证明不会删去合法key、纳入future key或重复sink。未完成任意Q、causal mask、prefill、verify、decode和graph变化验证前，不接入完整模型或宣称等价。
