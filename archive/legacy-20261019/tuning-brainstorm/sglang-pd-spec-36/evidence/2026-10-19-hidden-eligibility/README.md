# M016/M031：hidden 必要性、三步参考配置与单步条件

**本轮撤销“参考服务使用单步 EAGLE3”的陈述。** 原始进程记录为 CLI `NEXTN / steps=3 / top-k=1 / draft=4`；另一份 server-info 独立记录 steps3 / draft4。[参数勘误与原始文件 hash](reference-flags-correction.json)保留来源。当前源码的非 Gemma NEXTN alias 返回 EAGLE；没有重新读取远端当前规范化后的配置。L028 的 EAGLE3 CPU 夹具不是运行态配置证明，其原始组件结果不变。

因此，单步 hidden 省略不能作为这份三步参考配置的等价优化。本轮不实现或接受删除 hidden 的补丁。L028 旧说明同步更正，旧 commit 保留历史；原始测试数据没有改写。

## 固定源码与协议前提

固定 `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的 [18 个公开 Git blob](source-index.json)：17 个本轮核验，另一个 FutureMap 文件复用 L028 同 commit 校验。公开源码恢复不代表远端当前运行状态已复查。

| source chain 参数 | 原始 helper 的结果 | 含义 |
| --- | --- | --- |
| topk1，steps3 / draft4 | 有效 | 与历史参考参数相符 |
| topk1，steps1 / draft2 | 有效 | 不同配置的条件实验 |
| topk1，steps1 / draft4 | helper 断言拒绝 | 不能保持 draft4 便把三步称为单步等价删工作 |

见 [chain buffers](sources/installed/python/sglang/srt/speculative/base_spec_worker.py#L115)。这是 helper 合同；不等于断言所有 CLI 入口都会原样拒绝，该固定源码的 topk1 参数 hook 会把 draft 数规范化为 steps+1，见 `speculative_hook.py:1092`。改变步骤数会改变 proposal / acceptance 条件，必须作为配置对照，不能混称同工作量优化。

## 120 项 / 21 场景：完整源方法，模型边界明确

[R003 结果](hidden-cpu-result.json)载入完整 chain-buffer、选择 helper、IndexTopKShareState、draft_forward / idle、NextN forward 和 resolve 方法，使用真实 CPU Torch 2.8.0 tensor；模型、投影、探针和 context 等边界明确。

- 条件 steps1 / draft2：两/三请求、rejection proposal 有无、hot map 有无共 8 组 hidden 与 None 对照。所测 draft_forward 输出相同；idle 单步返回空拓扑。没有覆盖 prepare / ForwardBatch / graph / PD 或模型输出。
- 该方法在最后一步选择后 break，单步不会进入其中的 model.forward。不能据此推导“原生没有其他工作”：prepare 仍构造 cache loc 和 ForwardBatch，实际 backend / capture 仍需检查。
- 历史参数对应的 steps3 / draft4：两/三请求均到达第一个模型输入边界，hidden 是否缺失会传入该调用。实验在边界停止，没有完成原生模型，也没有测试三步生成输出。
- 实际 NextN forward 源路径在 projection 前读取 prior hidden、归一化并拼接。实验的 CPU RMSNorm / projection 边界对 Tensor 输入成立，对 None 拒绝；不是实际训练权重或 NPU projection 验证。
- `need_hidden_states=False` 本身不会清除已存在的 raw hidden；消费者源方法只在 gather 返回非 None 时替换它。不能假定改一个 gate 就让后续不再读取 hidden。

见 [draft_forward](sources/installed/python/sglang/srt/speculative/eagle_worker_v2.py#L779)、[NextN 输入](sources/installed/python/sglang/srt/models/deepseek_nextn.py#L194)。CPU 指针检查还观察到 `repeat_interleave(1)` 产生新 storage；这不是 NPU launch 或耗时测量。

## 22 项 / 4 场景：filter → resolve → NextN 输入

[R004 结果](filter-hidden-result.json)运行三个完整原始方法。survivor 索引是输入夹具，未执行完整请求取消；gather 使用明确 CPU oracle，norm 使用 CPU RMSNorm，投影停止。

| 从 3 行过滤到 2 行 | future-mode filter 后 | resolve 后 | NextN 输入路径 |
| --- | --- | --- | --- |
| 原始 hidden relay 开启 | 仅更新 future 索引，raw hidden 暂留3行 | 按请求键重取为2行 | 到达 projection 边界 |
| 显式构造的 relay 关闭状态 | 同样暂留3行 | hidden 没有被覆盖，仍3行 | 在 CPU 拼接前后形状检查中拒绝 |

两种保留顺序 `[2,0]` 与 `[0,2]` 均验证。它是对“直接关 hidden relay”的反例，**不是当前未修改服务发生形状错误的证据**。原始开启路径在该 CPU 合同中正确对齐。只保留字段名或旧 tensor 引用，不能替代请求行与生命周期合同。

## graph / adaptive 与 PD 限制

draft-decode graph capture 受 steps>1 条件控制；adaptive 会构建并切换不同 SpecRuntimeState。当前步恰为1不意味着下一轮也为1，更不能据此丢掉下一轮多步需要的状态。[graph gate](sources/installed/python/sglang/srt/speculative/eagle_worker_v2.py#L566)、[adaptive state](sources/installed/python/sglang/srt/speculative/adaptive_runtime_state.py#L122)。

ForwardBatch 的 draft padding / unpadding 还保存 hidden 引用并读取其 shape；draft-extend、PP、mixed、PD 首批 metadata 各有自己的消费者。这里做了源码范围审计，没有逐个执行原生路径，也没有建立可用于删除字段的完整准入证明。

## 决策与下一步

本轮 **PIVOTED**：历史三步参考配置不满足“单步省略”的前提；停止把直接删除 hidden 当该配置的优化。单步1/2的条件 CPU 等价不能替代三步3/4模型正确性和同条件性能。

hidden 的信息仍需正确到达多步模型。后续若研究成本，应先区分必要字段与冗余复制/选择：topk1 的选择可能保持请求行恒等，CUDA 已有部分链 shortcut，NPU 路径仍需逐项确认别名、in-place、graph 和 fallback 前提。不能从 CPU storage copy 推导实际 NPU 收益，也不因一个局部恒等关系删除状态传递。

新增候选若采用引用复用，会延长 buffer 生命周期并增加 batch-key / generation / fallback 检查；若改变步骤数，则改变接受分布。行对应、输出或生命周期任一不一致即否定。没有暴露成本与同条件测量，不填写加速比。

[原生合同](native-next-contract.json)仍要求任务自有 Ascend、PD 分离和推测解码/MTP，重新记录实际规范化参数、成功小 native allocator、真实模型 / gather / 流 / PD 回收 / 输出和性能。全部36机制目标未完成，整体 **E0**、accepted baseline 为 null；无生产修改、接受候选或原生速度结论。
