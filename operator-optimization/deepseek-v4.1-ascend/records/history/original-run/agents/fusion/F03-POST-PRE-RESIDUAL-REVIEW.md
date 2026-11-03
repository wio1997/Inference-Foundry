# F03 补充：attention HC post → FFN HC pre 的残差复用边界

> 更新：stage119 已取得两容器一致的 HC 源码；有关归约轴、内核同步及外部 pre_mix 的结论，以 `F03-HC-SOURCE-CONTRACT-AND-TILING.md` 为准。该快照尚未绑定运行二进制。

## 为什么优先看这里

既有 `agents/diagnosis/non-mc2-operator-aggregate.json`（输入 `captured-layer-attribution/operator_metrics.csv` SHA-256 `2a8d39f101c7e8949931390d49a3b2dcce58aee13bfe0bf799f789b81f8f55f4`）覆盖 16 rank × 3 个内部采样轮 × 40 层：归属为 `HcPre` 的任务 3840 次、耗时和 135.207 ms、单次平均 35.210 μs；`HcPost` 3840 次、耗时和 29.352 ms、单次平均 7.644 μs。它提示 HC 值得检查，**不证明可融合这两类任务、不证明任一相邻配对的节省量，也不是端到端耗时**：不同 rank 和流可并行，上述任务名可能对应多个调用点，归属规则不是逐 tensor 地址绑定。`HcPre` AIC MAC 代理约 5.74% 也不是整个 NPU 的利用率或低效的充分证据，HC 包含非矩阵工作。

这次深入的是原 F03 中**同一层 attention post → FFN pre**，不把既往针对下一层的 `candidates/F03-G2.json` 当成完整覆盖。已冻结 `model.py` SHA 与 `agents/fusion/frozen-configuration-evidence.json` 锁定一致，以下行号相对 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py`。

## 精确消费者与已有复合接口

`879–920` 的同层实际源码依赖如下，变量是源码调用语义，不等于已经抓到的张量地址：

1. 首个 `hc_pre(hidden_states, hc_attn_fn, ..., pre_mix)` 给出 `x`, `attn_post`, `attn_comb`, `attn_pre`。`x` 经 input norm、可能的 SP all-gather 与 self-attention；SP 模式先 reduce-scatter attention 结果（`896–901`）。attention post 接收该结果、原始 residual、`attn_post` 和 `attn_comb`，生成新 hidden（`902`）。
2. 这个新 hidden **有两条显式数值消费**：一条进入 FFN `hc_pre`（`905–911`），另一条被赋为 `residual`，并在 FFN 结束后进入 FFN `hc_post`（`904,919`）。`attn_pre` 也直接进入 FFN `hc_pre`；它来自更早的 attention pre，不能只拿 attention post 的输出便自足地开始 FFN pre。
3. FFN `hc_pre` 的 `x` 给 `rms_norm_cast`；norm 的 BF16 主干和 FP32 路由输出共同进入 MoE（`912–917`；具体设备接口与 fallback 的数值差异见 `F20-NORM-ROUTER-CONSUMERS.md`）。`ffn_post/ffn_comb` 等待 MoE 输出后才进入 FFN post；`ffn_pre` 与 FFN post 的 hidden 一起作为层输出（`919–920`）。
4. `hc_pre` 本来就是 `npu_hc_pre_v3` 复合调用，`hc_post` 本来就是 `npu_hc_post` 调用（`858–877`）。Python 级两次调用不能证明这两个算子内部结构、物化位置、硬件层次或编译 lowering；未找到同一冻结源码集内的两个 kernel 实现，不能把 pre 内部重复当成待融合的独立任务。

## 可以研究的重组与不能跳过的东西

**候选 A：attention post 的行/块输出直接供 FFN pre，同时为 FFN post 保留 residual。** 分块单位必须由真实 `hc_post` / `hc_pre` 归约与输入契约确定。可考虑产生一个块后，向 FFN pre 供给它，同时以可观察等价方式提供 FFN post 之后还要读的同一 residual。由于第二消费者隔着 norm 和 MoE，不能基于“pre 已消费”删除 residual，不能直接宣称原隐藏张量写和两次读均可省；如果其保留于片上，也需证明容量、生命周期、异步安全和后续 consumer 可访问。仍需 `attn_pre`、FFN 的权重/偏置等完整输入；pre 返回的 `ffn_post`, `ffn_comb`, `ffn_pre` 及 x 都要保留。

**候选 B：A 与 FFN norm 三段组合。** 在前一候选正确实现后，可研究 FFN pre 的 x 直接供双输出 norm；router 的 FP32 和 expert 的另一输入均必须按实际分支语义提供。现有 norm 可是 `npu_rms_norm_cast` 或 fallback，不应再次把它内部的 norm+cast 作为新增融合。应保持 pre 输出和 norm 输入之间现有 dtype 舍入、整行归约、epsilon、scale、权重，以及非有限数语义；不得默认 fallback 和设备复合算子逐位一致。若 x 真实有一次独立写和读且可被消除，其条件逻辑流量按 `2 * bytes(x)` 记账，并非测得 HBM 流量或保证节省，不能与已可能被编译器消除的流量重复计算。

**候选 C：跨层 FFN post → 下一层 attention pre。** 原 F03-G2 的条件直接边只在中间不触发 Engram 修改时成立；`model.py:1556–1592` 的 aux hidden 在可配置层取前一层残差的 mean，再可能先进入 Engram 修改 hidden，之后才调用当前层。末层还有 MTP hidden buffer、collapse 和 norm 消费（`1593–1605`）。因此不可假设所有层的 post 输出只有下层 pre 一个消费者，也不可跨越 aux 观察点或 Engram 原位写。配置推导与真实执行仍分别标识，不声称每层都进入这些路径。

**数值与时序限制。** 禁止把 post 和 pre 的非线性/归约换成未证等价的线性矩阵组合；严格保留当前两个算子之间 dtype 舍入、归约次序、Sinkhorn 迭代和所有输出语义。不能在 attention 的 SP reduce-scatter 对所需输入准备好前消费，也不能提前释放或覆盖 FFN post 所需 residual、系数和 MoE 输出。需核实 graph stable buffer、别名及事件；即使两个块无数据依赖，Cube、Vector、MTE 的争用也可能使重叠无收益。

## 未知和下一个有信息量的动作

历史样本中各 `HcPre` / `HcPost` 是否分别对应该同层边界、两个 custom-op 的输入输出 dtype/地址、内部计算和跨 token 依赖、片上容量、图编译是否已融合、相对端到端关键路径暴露，以及该重组的正确性和收益均未知。应优先只读取与冻结运行镜像匹配的 custom-op schema/实现及其算子文档，并查看已保存 profiler 中是否存在可关联的节点与 storage 标识；没有这些材料就保持 UNKNOWN。无需为这条候选启动新的 NPU 测试。本次只做本地源码与既有采样审计，没有改框架代码或设备。
