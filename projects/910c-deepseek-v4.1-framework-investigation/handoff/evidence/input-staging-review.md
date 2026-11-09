# 输入 staging 所有权与提前交付判断

**Goal ACTIVE；全链路尚未扫全；每个替换原语必须逐位一致。** 此段是源码所有权判断，不是设备实现或性能验收。范围为当前 V1 / DSpark eager、TP8 DP2、CANN9.2.0 的输入到 H2D 末读；旧模块完成标志不作为覆盖证据。

## 介绍

提前只读 D2H 的价值来自 target IDs 已可用而 DSpark 尚未结束。上一轮已排除“pinned H2D 自动快照”，本轮继续识别哪些物理源真的需要独立代次，避免先复制全部输入。输出 data-ready、CPU staging 可写、device metadata 可重用和外部 KV 退役必须分清。

## 到了哪里

[身份](staging-execution-identity.json)核对21份运行源码，当前 hash 一致；新增取得的5份源包括实际 NPUInputBatch、上游 block table 与 PIN_MEMORY 链。SHA 数只是身份，不是源码覆盖率；[前沿](staging-execution-frontier.json)列出实际读过的行段、6组物理来源、守卫与未确认项。

- **已有 CPU 保护不可漏算。** GPUModelRunner:788–793 在 async 时创建 blocking `prepare_inputs_event`，3841–3853 的 context 在进入时 host synchronize、退出时 record。Ascend execute_model:2288–2514 包住 `_update_states`、block table commit、输入和 attention metadata 准备；dummy:4045–4180 也参加。由此，target 这一段持久 CPU 源已有末读→下一次写的保护，不能笼统说所有源都无所有权。
- **block table** 的 `cpu/np` 共存储，append/add/clear/move/swap 修改行，commit 复制前 num_reqs 的完整已分配行宽。zeroer 的 INT64 `_ids_pinned` 也是持久前缀覆写，但其 `_update_states` 调用在上述 scope 内。原 a2 仍证明后端不会自动快照，不能据此声称未修改服务的这两条已保护路径发生 race。
- **已找到的晚到持久源** 是 proposer backup：上游构造 INT32[max_num_seqs=4]，每卡容量仅16B；Ascend proposer:2214 CPU 填写、2217 H2D，在 sample/bookkeeping 之后，不在 target scope 的这次事件内。关键是代次关系，不是这16B的带宽。
- **CP 元数据不可一概算成共享可写源。** dsa_v41_cp:118–140 的 qsl/suffix 为本次 clamp/subtract 结果后 pin；审阅路径未再原地修改。DeviceMetadataExecutor 的 stage/reuse event 则保护 device buffers。TargetDeviceMetadata.finish 会 join producer；这些 device 依赖不能替代 CPU lease。每次 build 的 shared 字典也不是已经证明可跨迭代复用的内容 cache。
- **较小的数学化简已追到消费者。** 当前 greedy K>0 的通用 verifier 至少写首个 target argmax，K=0 写 column0 bonus，K1 特化也写 column0；ID在词表内。无 discard 时，clone/index_fill 不会清空行，所以 valid_count≥1，`torch.where(last_valid_indices != -1, selected, backup)` 的 backup 臂无活读。可仅省去 fallback CPU fill/H2D，保留原 where、计数和其余原语。它作为提前输出协议的一部分继续判断，不把16B复制单列成有意义的加速。

两份原 C8 回复 worker 日志已重新 SHA 核对：rank0共2641条、rank8共2677条。[字段记录](staging-archived-eligibility-schema.json)没有 discard 数或 all_greedy 标志。仅看 scheduled 请求和客户端非空输出，不能证明同批每行都无 discard。没有为此重跑服务或收集 profile。

## 结果

| 范围 | 状态 | 已判断与具体缺口 |
|---|---|---|
| 原样只移动 D2H 的旧补丁 | 已停止 | 继续保留为失败设计记录，不能应用；没有晚到源的复用协议。 |
| 提前输出 + 扩展输入复用栅栏 | 待确认 | 可在最后 H2D/必要 join 后重录 reuse event；保守 draft-end 版本不必先建全部输入 ring，但须扣下一输入 CPU 准备失去的重叠及 driver 竞争，不能免费保持原吞吐/response end。需确认跨流最后读者。 |
| 提前输出 + 无 discard greedy 的 dead-backup 化简 | 待确认，值得深入 | 上述条件源码蕴含已推导；仍需全 caller/委托读者、prefill/dummy/空DP守卫、实际负载可用请求集合、native逐位与集成。不是已确认可实现。 |
| 有效 fallback 的独立 snapshot/ring | 待确认 | 仅给实际活的晚到物理源加lease，最后DMA读完才释放；算 allocation/copy/event/backpressure。没有证据要求所有 target 输入双份。 |

原7–15ms是保持 CPU handoff 和旧 copy/event 后的**重叠组件预算**：短 slot7为15.238845ms、3.3951% TTFT，整批P90为8.01552ms、0.5373%。本轮没有新增加速数字。无 discard 方案的实际可用请求集合尚缺，故不能把上述数值全额给它；栅栏方案还需扣 host 提交流水变化。没有认领吞吐、响应结束或六个普通负载的实际收益。

本轮新NPU工作0、模型加载0、服务推理请求0、新profile0、远端写入0。源身份只读和两份旧日志字段检查均在 /data/tiankuan/wio 冻结证据范围；本机只写本项目。分析器首版把 AST 遍历次序当源码顺序而失败，error Run保留；按行排序后审阅通过，这不是候选正确性通过。

下一处进入**共享 metadata / host submission 的较大交付**：结合实际 CPU DP 协调、目标 metadata producer、draft 准备和下轮提交，先定父范围和已有 profile 预算；本页留下无 discard 可用性与完整末读具体缺口，继续走剩余路径，不重复自动快照实验。其后回到末层 attention/FFN 的 row-demand producer。
