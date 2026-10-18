本包用于审阅下一次取证的范围和阻碍，**不是已完成的设备探针，当前不应申请新设备窗口**。`review-plan.json`的schema故意不同于launch schema，`execution_ready=false`；不能交给`start_capacity.py`启动。本轮没有SSH、资源申请、模型加载或设备取证，也没有新增同步、kernel或优化器。

先解决一个较小问题：用现有无NPU image采集入口核对18个源码文件的实际SHA，解除factory/proposer/executor目前只有固定提交参考的身份限制。`image_identity.py`只有标准库文件读取，不导入torch/vLLM，不观察实际proposer对象或任何KV内容。Mac上的同源fixture通过，不等于镜像配对通过。当前授权不含SSH，以下远端命令仅为未来重新授权后的模板：

```sh
python3 glm5-3-vllm/tools/collect_node.py --node 32 --image \
  --script glm5-3-vllm/diagnostics/p-first-state-v1/image_identity.py \
  --label "${FRESH_RUN_ID}_identity" --output "$NEW_LOCAL_EVIDENCE/node32-source" \
  --control "$NEW_SSH_CONTROL" --timeout 90
python3 glm5-3-vllm/tools/collect_node.py --node 33 --image \
  --script glm5-3-vllm/diagnostics/p-first-state-v1/image_identity.py \
  --label "${FRESH_RUN_ID}_identity" --output "$NEW_LOCAL_EVIDENCE/node33-source" \
  --control "$NEW_SSH_CONTROL" --timeout 90
```

这里没有`--npu-devices`，不占32/33的16die；镜像仍须符合collector固定image ID。上述脚本不运行本机实验，源码不匹配即退出2、保存差异，不自动修补/重试。不能把identity成功写成实际运行时proposer类型已经确认。

目前查到的完成边界与限制如下。逐方法SHA和行号在`evidence/20261018-p-first-state-audit/probe-preparation-v1/boundaries.json`。

| 边界 | 源码支持的结论 | 现场仍须核实 |
|---|---|---|
| P `sample_tokens` | drafter调用后才finalize connector；但Mooncake的`wait_for_save`为空实现 | 不能把finalize叫作MTP完成fence |
| P `AsyncGPUModelRunnerOutput` | 构造时copy stream等待当前计算stream；原生`get_output`等待已有copy完成event后才返回 | 实际MTP是否运行、其writer stream是否被这条依赖覆盖；CUDA兼容入口是否确为NPU event；不能另外全局同步后才判断 |
| P executor聚合 | 新补固定提交中，KV aggregator存在时向全部worker取输出，每个Async输出先执行`get_output` | 新文件尚未镜像SHA配对；实际aggregator、响应rank集合和每rank完成记录未捕获，不能只看rank0 |
| D接收线程 | `_handle_requests`等待transfer futures，bucket调用`batch_transfer_sync_read`；成功返回后才入finished队列，之后scheduler才能晋升 | 外部TE同步调用的实际实现/完成契约、失败集、所选P rank及各group地址；方法名称不能替代最后行内容证明 |
| D取样候选点 | 成功`_handle_requests`返回、finished入队之前，可避免本请求原生D query重写N−1 | 这里尚未实现接收线程与真实tensor views、block maps及请求的取证适配；不能在D首forward后补拍 |

需要区分三种情况，不能统称race：没有执行P MTP首轮；已发起但原生等待前仍未完成；原生等待依赖覆盖后已完成。前两者可用实际drafter分支/K、writer位置和stream、已有event在原生wait前后的状态区分；event尚未完成但随后被原生wait正常覆盖，并不是传输race。P HTTP返回→同一客户端发送D是可记录的因果边；不同节点/进程的时钟数值不能直接排序。新增读回只允许位于已经成立的原生完成边界之后，不能用新增synchronize、延时或barrier把缺失依赖“修好”。

如果实现后续小适配器，以下是验收条件，而非本包声称已经采到的字段：

- 每rank记录实际proposer类型、相关模块路径/SHA、方法/K和P MTP首轮确实运行的证据；保留已有完成事件与stream依赖。P样本在原生结果完成之后、对外暴露blocks之前获得。
- 按实际注册tensor view记录layer名、target/Indexer/MTP归属及共享alias、group/spec、dtype/shape/stride、page/block scale、逻辑位置到物理块/偏移。不能把packed主缓存、RoPE空view、Indexer K/scale或共享页统一按裸地址加固定offset解读；不支持的布局直接停止。
- P参考由本请求实际writer的slot/position范围与原生完成依赖确认，不以“非零”或allocator块内有值代替。MTP的`y1+h(prompt最后)`行仍只是固定源码预期，必须与实际执行位置配对。
- D依据真实transfer bucket所选的P PP/PCP/TP rank及层/region映射配对，不能默认P/D同rank或同物理block ID。两端使用同一prompt SHA、cache salt和remote request ID。独立salt保证两请求cold，禁止借旧key/旧metadata。
- 只保留诊断所需的行与摘要：N256取0/127/128/255；N257再取256。逐行SHA必须含dtype、形状、逻辑位置及对应字节。另须有覆盖全部有效前缀的writer/transfer范围和完整前缀摘要，不能由4/5行相等宣称整个MTP前缀有效。前缀摘要的读取与配对尚未实现；现有四个整数moment工具不是密码学字节一致性证明。
- D成功READ后、finished入队之前取样，比较对应P参考；任何缺组、缺component、错位、缺writer provenance、完成依赖不明或不完整前缀，结论均为无法支持直接推进并停止。不能用D重算后的正确内容掩盖原状态。

这至少需要两个原生边界的关联，以及一个经过布局/共享页和写入位置核验的行/前缀摘要适配器；现有MemCache元数据探针不能直接用于Mooncake，`start_capacity.py`也须随探针审阅新增的明确overlay目标。当前没有经过核实的tensor切片和异步生命周期适配，故未实现设备插桩。**建议先完成无需占卡的源码身份核验，再决定是否承担上述适配工作；当前不要为此重开32die模型窗口。** 若适配成本扩大到修改算子、通用缓存解码器或同步协议，应停止此候选。源码没有否定候选，但尚无整请求target forward减少的实测收益来支撑更大投入。

后续适配器就绪后，设备范围限定为32/33各16die、一次P/D加载、串行cold C1两请求N=256/257、P K1/D K5，原生协议，不采纳y1为新history、不启用发布overlay。`review-plan.json`中有精确token IDs、输入SHA、位置、基线配置SHA和采样预算。N257仅在存档256输入尾部追加同一token，用于非对齐块边界；不作为语义精度题。P/D首token采样均min_tokens=1、ignore_eos=true，D只输出2个token，仍须实际核对首ID及usage，不据配置宣称采样等价或完整精度。

首批6→6只说明首批未删，不证明整请求绝不可能少一次target forward。本轮两输出的原生取证不比较新组织的完整工作量；若以后进入推进query的候选实验，是否省迭代最终须按同预算、同采样的完整请求实际target前向次数判断。

预计资源时长须按完整启动计算：存档P11从docker launch 17:32:54.320Z到API ready 17:43:21.213Z为626.893秒，D12从18:40:20.046Z到18:50:56.688Z为636.642秒；仅模型开始加载到ready约201/213秒会漏掉约7分钟。并行加载约11分钟是历史参照，不保证ETA；就绪、两请求、取证与保全合计估计18–22分钟，建议最大窗口25分钟，不续期。若15分钟仍未就绪或首请求映射不完整，停止本轮并保全失败证据。这个预算目前不构成窗口申请。

真正可执行前须生成新授权记录、两枚fresh lease UID、owner/node校验、新run目录/容器名/端口和绝对截止时间；不能复用本包引用的旧launch配置的lease/CID/lifetime，也不能用D13。完成上述适配与身份核验后，才有资格生成传给旧启动入口的新plan：

```sh
# 模板：APPROVED_DIR中的文件当前不存在，也不是review-plan.json。
python3 glm5-3-vllm/tools/start_capacity.py --plan "$APPROVED_DIR/P.launch.json" \
  --control "$NEW_SSH_CONTROL" --output "$NEW_LOCAL_EVIDENCE/P-launch"
python3 glm5-3-vllm/tools/start_capacity.py --plan "$APPROVED_DIR/D.launch.json" \
  --control "$NEW_SSH_CONTROL" --output "$NEW_LOCAL_EVIDENCE/D-launch"
```

成功、失败或时间截止后均先保全两端新plan/源码SHA、每rank关联日志、P/D映射与摘要、请求/响应、原生完成与首D forward顺序、停止日志及失败原因，下载Mac核对SHA；仅停止本次launch receipt精确CID并核对名称/镜像/运行目录。旧`stop_capacity.py`硬编码历史实例，不能照用。核对退出和两台NPU进程归属后，以新holder的精确UID为删除precondition释放自己的预留并复核不存在；不触碰别的作业，不以延长lease等待归档。保护好证据和失败状态后停止，不自动重试/扩大测试。

Mac核验命令：

```sh
python3 glm5-3-vllm/diagnostics/p-first-state-v1/validate_packet.py
python3 glm5-3-vllm/diagnostics/p-first-state-v1/check_local.py
```

语法、18项同源fixture身份、空/错SHA拒绝和case位置/缺项拒绝结果存于`evidence/20261018-p-first-state-audit/probe-preparation-v1/local-checks.json`。这些检查只验证准备材料，不是设备取证运行通过。
