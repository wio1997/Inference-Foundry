# 第 0 层及热点：三个融合边界的源码复核

这三项是**条件检查对象**，不是已实现的新算子或已证明的提速。第 0 层是已有 48 个捕获 target 调用的同口径样本里活动代理最低者；其中 MC2 占该层已归属 AIC 周期分母约 81.7%。该计数不是端到端时间，也不能证明以下边界就是低活动的原因。见 `evidence/captured-layer0-engine-diagnosis.json`。性能排序仍未知。

## 1. F02：attention mHC pre 的混合值 → 输入 RMSNorm

**生产者和已知消费者。** `model.py:888-902` 令 `hc_pre` 返回混合输入 `x` 和 `attn_post/attn_comb/attn_pre`；`x` 的源码显式数值消费者是 `input_layernorm`，归一化结果在 SP 分支还要 all-gather，然后给 `self_attn`。同一个 `hc_pre` 的 `attn_post/attn_comb` 给 attention 后 `hc_post`，`attn_pre` 给 FFN 的 `hc_pre`；最初的 residual 也供 attention 后的 `hc_post`。这些是不同输出，不能因为只融合 `x→norm` 而去掉。

**可能少什么。** 如果按实际 dtype/shape 确认中间 `x` 本来需要物化，且融合实现能保存相同的中间舍入，那么可讨论省掉该中间 `x` 一次写和一次读，符号表达为 `2*bytes(x)`。它不是已测 HBM 流量、保证节省量或性能预估；若优化器已消除物化或数据位于片上，实际差异不同。

**必须保持。** `hc_pre` 的其他输出和 residual；RMSNorm 完整归约与原 dtype 舍入；SP 的分片/all-gather 边界；attention 独立 Q/KV 流、KV 缓存更新、跨流事件。空 query 或 graph capture 分支不能被无条件省略。`npu_hc_pre_v3` 本身已是复合调用，内部算子不能再按完全独立计费。

**未知。** 层 0 这次捕获中内部真实物化与 dtype、图 lowering、算子支持的精度保持和片上容量；该候选是否改变关键路径或破坏现有重叠均未知。`F02` 已有结构记录，不证明能实现。

## 2. F05：Q norm → 动态量化 → Wq_b

**生产者和已知消费者。** `dsa_v41.py:320-324,352-384,440-495` 中，Q norm 产生 `qr`，主 Q 路径消费 `qr`，经量化/Wq_b 生成 Q 并 RoPE。`_prepare_indexer_inputs` 只在 long-context 且 index-source 时建立 indexer 输入（`dsa_v41.py:393-395`）。冻结角色图层 0 的 compression ratio=0，且不是 index source，因此该路径里**没有同层 indexer 对 qr 的数值消费者**；其他 source 层则不同，不可把这一点推广到全部 40 层。`qr` 后续还可能用于形状元数据等；不得假设全部别名和 lowering 已证实。

**可能少什么。** 在 `qr` 是 BF16、宽度为 1280、每 rank 有 M 个有效 query 行且确实物化的条件下，`qr` 一次写加一次读的候选上限为 `2*M*1280*2=5120M` 字节；这是候选逻辑流量，不是必经 HBM 下界或节省量。若继续接入 GEMM，还要重新审计 INT8 输入和 scale 的复用、分块及容量，不能直接累加。

**必须保持。** Q norm 的 BF16 舍入、动态量化行范围/scale/取整、Wq_b 权重和输出、RoPE；已有代码把 Q norm/quant 与 KV matmul 在不同流交错，且最终 join 保证 SWA cache 写对 attention 可见。不能融合后直接删除 KV cache 写、消除必要 event 或假设 Q/KV 的行域在 CP 模式一致。空 query 可能只更新 cache，不可要求这时必定存在 Q 运算。

**现有数据及未知。** 16 rank 1920 个已归属 target 层同形状 Wq_b 样本整体中位数 42 微秒，是 40 层的汇总，不能当成只测量第 0 层，更不能当成此候选能节省的时间。`evidence/q-projection-16rank-diagnosis.json` 中 118 倍只是受限假设下的敏感性算术，不是可实现加速。实际计算/量化实现、融合等价、物化、资源冲突、端到端影响均未知。

## 3. F23：MoE 输出缩放/合并 → FFN mHC post；并审计 F21 已有 MC2

**生产者和已知消费者。** `model.py:262-328` 的 MoE 中，expert 返回 tensor 或 tuple；tuple 分支可执行已存在的 `muls_add_triton`、缩放或必要的 TP reduction；tensor 分支由上游 runner 完成相关归约。`model.py:904-920` 将最终 `x` 送进 FFN `hc_post`，后者同时读取 attention 后残差 `residual`、`ffn_post`、`ffn_comb`。FFN `hc_pre` 的 `ffn_pre` 独立返回给下一层。第 0 层产出的完整 HC hidden 还进入第 1 层；该层的 Engram 条件存在时先通过 `layer.engram` 修改 hidden，然后才调用第 1 层。不得将全部输出当作 MoE 单一内部临时值。

**可能少什么。** 可条件地检查把当前 MoE wrapper 的最终输出与 `hc_post` 接合，理论上避免该 `x` 一次物化写入和读取，最多 `2*bytes(x)`；实际可能没有独立物化、或需要通信/缓冲，不能当作已节省的 HBM 流量。现有 `muls_add_triton` 已经融合相关加法/缩放；F21 的 fused MC2 已包含 dispatch、GMM1、激活/量化、GMM2、combine，**不能把这些阶段再次记作新融合**。只有选定执行分支真正存在缩放/合并时，该子边界才存在。

**必须保持。** router FP32 和 expert BF16/INT8 数值、scale/round、top-k 并列、通信组与 MoE 输出归约、全部通信完成及 readiness；`hc_post` 的 residual/post/comb 数据与 `ffn_pre`，以及 Engram 条件状态、下一层读取。rank 间到达不齐并非可靠相邻算子融合直接认定解决；设备时间可能含等待。

**未知。** 第 0 层样本实际 tensor/tuple 路径和 shared expert 分支、输出通信完成点、内部专家负载与 MC2 等待因果、全部层间缓存 alias、候选收益。源码允许的融合不证明芯片接口支持或精度等价。

## 检查和恢复

源码 `model.py`、`dsa_v41.py` 与配置来源分别由 `agents/fusion/frozen-layer-coverage.json` 和 `frozen-configuration-evidence.json` 中 SHA 锁定；本报告使用 `evidence/QR-consumers-and-fusion-review.md`、`evidence/captured-layer0-engine-diagnosis.json`、`evidence/q-projection-16rank-diagnosis.json` 和 `evidence/MC2-known-dependencies.md`。这些原有记录须与最终结果一起同步，不能将单个条件流量当成实际收益。
