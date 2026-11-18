# 完整 D 图的 MoE 路由门槛

G373 的完整 target/DSPARK 权重装载后，D 图实际经过先前的 RoPE 与 mHC 调用位置，在 `deepseek_v2.forward_deepep → TopK.forward_npu → fused_topk_npu` 的 sqrtsoftplus 分支报错：固定 CANN 9.1 缺少 `aclnnMoeGatingTopKHash`。D 控制器终态为 failed_released，8卡进程已清空。没有通过缩小图、关闭 DSPARK、改量化或升级镜像绕过该缺口；完整 P/D、模型质量及性能仍未验收。

G385 复用同一镜像已有 `moe_fused_gate` 融合 Triton 路由。8 rank 的空行接口可返回，但第一次非空输入编译在 CUDA `libdevice.log1p/exp` 处失败，没有非空数值或图通过。这轮失败按原样保留，不能称为该实现已适配 NPU。

G391 候选只替换内核中的三种数学调用：exp/sqrt 使用 Ascend Triton 支持的 tl 操作；log1p 用原文件稳定分支已有的补偿公式。AST 核对确认其余整个融合内核保持原逻辑，包括 V4.1 的 NaN 排名分支、偏置排名、top-k、归一化、shared slot 和缩放。原生 Python wrapper 的 code object 保留，并给候选内核提供独立 globals，未覆盖运行包文件。

候选组件覆盖 target384/top6、draft128/top3、shared slot0/1、缩放关闭/开启、行数0/1/16/96/8192与结构化/极值/tie输入；排名参考按 FP32 契约计算，权重参考用 CPU FP64。组件结果另行验收；本次归档只保留源码候选和既有失败，不宣称新内核、完整图或端到端性能通过。

- [G373 原始完整日志](../evidence/full_model_G373/2204/server.log)
- [G373 D 终态](../evidence/full_model_G373/2204/state.json)
- [缺失算子位置及权重 ledger 原始 SHA 索引](../evidence/full_model_G373/2204/assessment.json)
- [G385 原始非空编译失败](../evidence/native_router_G385_failure/server.log)
- [G391 精确替换清单及 SHA](../code/native_router_G391_candidate/lowering_manifest.json)
- [G391 候选内核](../code/native_router_G391_candidate/route_kernel.py)
- [固定镜像原生融合路由源码](../code/fixed_router_G385/sglang/kernels/ops/moe/moe_fused_gate.py)
