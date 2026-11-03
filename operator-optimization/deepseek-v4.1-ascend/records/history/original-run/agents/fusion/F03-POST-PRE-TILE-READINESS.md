# F03：attention post 与 FFN pre 分块流水的就绪条件

> 更新：stage119 已取得两容器一致的 HC 源码；有关归约轴、内核同步及外部 pre_mix 的结论，以 `F03-HC-SOURCE-CONTRACT-AND-TILING.md` 为准。该快照尚未绑定运行二进制。

本文承接 `F03-POST-PRE-RESIDUAL-REVIEW.md`。检查对象仅为同一层 `model.py:879–920` 中 attention 的 `hc_post` → FFN 的 `hc_pre`，不是跨层 F03-G2；冻结 Python 代码可确认完整张量调用接口，未提供这些 NPU 算子的完整内部计算/存储/分块实现。下列是**成立条件和接口缺口**，不是已经证明实现可行、已经执行或性能成果。未运行设备。

## 一块需要什么

将一个候选块称为 B。它可以是若干 token 行、HC 分量与 hidden 特征的集合，但尚不能假设不同 token 行互不依赖，更不能假设 arbitrary feature 子块可以单独完成 pre。

| 阶段 | 源码明确输入 | 需要满足的依赖 | 结果和消费者 |
|---|---|---|---|
| attention post 的 B | attention 最终输出 `x`、层入口 `residual`、早先 attention pre 的 `attn_post`、`attn_comb` | 具体哪些元素/行够用取决于未取得的 `npu_hc_post` 内核；若 SP 为真，`x` 先来自 `sp_reduce_scatter`，不能在该 B 的所有必需跨 rank 贡献完成前消费 | 生成 post residual `R`；`R` 同时供 FFN pre 和稍后 FFN post |
| FFN pre 的 B | `R`、`hc_ffn_fn`、`hc_ffn_scale`、`hc_ffn_base`、先前 attention pre 的 `attn_pre`（作为 `pre_mix`） | B 所需 `R`、`attn_pre` 和权重在对应存储域就绪，且原 pre 所有归约/迭代依赖已满足；仅有一部分 feature 不自动意味着可产出完整结果 | 生成 `x_ffn`、`ffn_post`、`ffn_comb`、`ffn_pre`；四者都不能丢 |
| norm 和 MoE 的 B | `x_ffn`，norm 权重/epsilon；MoE 的 router 与 expert 输入、可能的 input_ids/通信状态 | `rms_norm_cast` 的归约域完成后生成双输入，再按实际路由/分片/通信规则执行；不能假设只凭单个块就能越过 MoE 的内部交换或全局元数据 | MoE 输出给 FFN post；此前产生的 `ffn_post/ffn_comb` 也要活到此处 |
| FFN post 的 B | MoE 结果、同一 `R`、`ffn_post`、`ffn_comb` | 同一块的最终 MoE 值及残差/系数均就绪；跨流发布、重用覆盖与实际 alias 必须安全 | 更新后的 HC hidden 与 `ffn_pre` 作为该层输出，受后续层/aux/Engram/MTP 消费限制 |

注意 `R` 的第二次消费跨过整个 FFN/MoE：让第一消费者边生产边读，并不意味着可以扔掉 R。允许的实现可以对原值做持久化、保持可访问的内部缓冲、或以证明等价的方式重建，但每种做法都必须满足原 dtype 舍入、别名、消费者和生命周期；不假设片上缓冲足够长久或跨 kernel 可用。`attn_pre` 同样需要从 attention pre 保留至 FFN pre，不能从 `R` 独立推导。

## 哪些归约和同步不能凭空隐藏

- **HC 内部**：构造参数 `hc_mult`、`hc_sinkhorn_iters`、`norm_eps` 和 `hc_eps` 被传入 pre（`858–869`），`hc_fn` 的声明维度是 `[(2+hc_mult)*hc_mult, hc_mult*hidden_size]`（`821–829`）。这些明确要求保留对应映射和迭代配置，但仅凭形状和名字不能恢复 `npu_hc_pre_v3` 的精确归约轴、迭代式、跨行通信或内部 barrier，也不能证明独立 token 可流水。对于任何实际存在的归约，块结果只在该归约所需所有参与值到齐后可消费；如果 Sinkhorn 后一步依赖前一步，就必须维持该迭代依赖。特征切块可以先做部分量，但不得把部分量视为最终归约结果。需要厂家算子契约或匹配实现确认确切域。
- **FFN norm**：`model.py:844–852` 按 `post_attention_layernorm` 权重和 epsilon 调用设备复合 `rms_norm_cast` 或 fallback。RMSNorm 的归一化依赖其规定的完整归约域；要分块处理 hidden 特征必须合并所需统计量后，才能按原算子语义输出相应结果。设备复合和 fallback 的实际归约顺序及双输出舍入是否逐位一致尚未证明（参见 `F20-NORM-ROUTER-CONSUMERS.md`），不得以一个分块统计近似替代完整域。
- **跨 rank 就绪**：`model.py:832–836,897–902` 在 SP 条件下先把 attention 输入 all-gather，attention 输出走 reduce-scatter，然后才作 attention post。声明配置有 TP8、DP2、EP 和 FlashComm；是否真实启用 SP 只能条件推导（`frozen-configuration-evidence.json`），不能把 collective 作为所有分支的已测事件。原源码呈现的是完整 collective 返回后整张量消费，**没有给出可独立等待/可读取的 per-block 完成事件**。因此“一个块到了就开始 post”还要求通信接口提供已完成且不会被覆盖的子块、所有参与 rank 的贡献齐全，以及一致的分块/collective 顺序；否则只能按完整 collective 完成推理，不能凭时间相邻推断已隐藏通信。
- **后续 MoE**：`model.py:262–328,913–919` 把 `already_sequence_parallel=use_sequence_parallel` 传给 MoE；其内部路由、expert 输出和 tuple/tensor 的归约分支不同。块化不能跳过任何需要的跨卡 dispatch/combine、输出归约、shared expert 合并，或让部分 expert 结果被 FFN post 当成最终值。源码不提供此次运行每一块的 rank 映射及可见性。
- **并发顺序**：必须确保该块依赖的写入先于读取，两个消费者结束前不可回收或覆盖 R、`attn_pre` 和 `ffn_post/comb`；在不同流/子块间需要足够细的 completion 与 reuse 信号。但现有 Python 前向没有列出该边界的独立 per-block event，不能声称存在或不存在底层隐式同步。相同数据依赖允许并行只是必要条件；资源竞争、全局 barrier 或设备调度可阻止有益重叠。

## 是否已有这条融合

源码 `858–877` 分别调用既有 `npu_hc_pre_v3` 和 `npu_hc_post`，`902–911` 在 Python 层显示连续的 post 和 pre 调用；前者和后者各自是复合接口，不应重新把其内部片段都记作未融合。现有冻结源码集搜索未找到这两个接口的底层 kernel 实现，也未在该模型前向发现专门的 `post+pre` 单一调用；这只说明**代码层未发现专门接口**，不证明编译后的图未融合或没有硬件流水。已有任务型 trace 的两个名称存在不证明 tensor 逐次关联或缓冲物化。

## 结论及最小资料需求

可提出一个**条件化**方案：当设备/通信接口能提供相应稳定就绪子块、HC 依赖域和归约结果得到满足、残差可安全保留到 FFN post 且保持所有输出和精度时，post 对某块的生产与 pre 对已完成块的消费可以尝试流水；现有材料无法证明这些条件成立或它能提速。进一步只需读取与冻结镜像一致的 `npu_hc_pre_v3` 和 `npu_hc_post` 接口/实现（尤其输入输出 dtype、别名、数学和归约轴、块内同步）以及 `sp_reduce_scatter` 的块就绪/流语义；若 profiler 本身保存精确 tensor 地址或图节点依赖，再以同 rank/graph replay 关联现存数据。缺失即保持 UNKNOWN，不需要新测卡。
