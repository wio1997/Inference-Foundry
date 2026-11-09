# 共享 metadata 与 host 联合交付判断

**Goal ACTIVE；loop-014 已 PIVOTED（限定结论），当前 loop-015 EVALUATING；全链路未扫完。每个替换原语逐位一致。** 本轮追实际协调与元数据消费者，未运行新 NPU、未部署。

## 介绍

范围从输入 scope 内的 target DP 协调、两套 global/local 元数据，到 DSpark query 准备、draft 协调、producer stream 和 model consumer。先确定字段何时可知，再决定哪些 Python 准备、复制或发布可以共享。CPU 等待 peer 的时间不等于可移除通信成本。

## 到了哪里

[当前身份](metadata-host-execution-identity.json)核对13份实际源码；[前沿](metadata-host-frontier.json)列出已读行段与合法条件。计数仅作身份，不证明全链路覆盖。

1. **CPU DP 实际拓扑与规则。** `parallel_state.py:2052–2058,2188–2204` 为 TP8/DP2 生成八组 `[r,r+8]`，CPU backend 是 Gloo；不是一次16-rank allreduce。runner:825–885 打包本DP count/mode，blocking all_reduce 后取 mode 的最小值，按角色、MoE/图条件决定 padding，target:3325–3432 再 dispatch。dense、KV consumer、hierarchy、MegaMoe 有不同守卫，不能直接按 DP2 删协调；冻结 target/draft 都是 MoE、无 KV consumer。
2. **未来 draft count 可先算，内容不可先算。** 实际类是 `AscendDSparkProposer`，其 override:226–357 返回 `B*num_query_per_req`；anchor=true、K5时为5B。GPU reject count 改 positions/seq_lens，不改这个 padded query 数。故 target24桶不能直接当 draft24：B4需20，B1需5。anchor=false为(K+1)B；dummy:366–429 另有min/cap，K0/skip/error/empty需保持collective顺序。显式扩展packet可提前携带两角色字段，但不使slot/length/position变成早知内容。
3. **同代共享边界。** runner:3822–3883 为每次 build 新建 batch 字典，每个 KV group 另有物理slot字典。CP global 的共享字典再分 `cp_global` 命名空间；local域的Q、seq_lens、position、actual计数不同。base builder先复用batch，再按资源kind/ratio/block table/slot/C2 state组装和发布。可共享BatchFrame，不能缓存整份返回对象后跳过每组副作用。
4. **实际 V1 producer 模式。** runner:6128–6139 创建 `DeviceMetadataExecutor()`，默认capture_producers=False；3900–3905 每次host submit，FULL传batch_descriptor做external event。`_model_forward` finally调用release。`TargetDeviceMetadata.finish` 的全task join/capture_producers=True属于别的路径，不能拿来证明本 V1 已 join 所有末读。需要继续追 V1 release 与未消费/empty-query task 的实际关系。
5. **只在推导边界后复用旧证据。** 重放旧 CP length-elision 与 CPU-plan renderer，精确得到 e79e503… / ef972657…，并与旧记录一致；21项profile输入hash全通过。[资格](metadata-host-history-qualification.json)。其中“避免25次无活读length计算”“25次CPU-plan命中”已是历史方案，不能作为新增收益。旧C8 CPU-plan相对它自己的CP+slot参照整批约10.364%/9.560%，但只验文本/选定metadata，**不满足本次每替换原语逐位门槛**。

原 joint168 C8 DB 未保存Python/PyTorch/OSRT调用区间，不能将空隙命名成Gloo。旧CPU residual cohort则有128个B4/T24/actual4/causal generation，每代26次CP调用、25次plan与batch命中；其每代完整host包络中位9.29759ms，包含诊断、提交与回压。112个native period与128代不是一一对应。此包络不是9ms净收益，不能加跨rank，也不能加已重叠的native union。它是不同、已做CPU-plan的诊断cohort，不是当前stock整批或普通负载的计时。

## 结果

| 范围 | 状态 | 判断与具体缺口 |
|---|---|---|
| 两角色显式 DP packet | 待确认 | 早交换target count/mode和B/Q/draft count、epoch/active/dummy资格，保留两套原规则；全体同意才省后一次协调，否则全体恰好回退一次。还缺empty/dummy/K0/input-fits/error序列与matched all-rank成本/peer-readiness。 |
| 同代global/local BatchFrame + 每资源组装/发布 | 待确认，值得深入 | 指定可共享视图/标量和仍必须保留的physical state。新增价值需来自剩余lookup/视图/config/组装/发布；先前CPU-plan收益不重复加。完整observer/alias、V1末读、native逐位、图地址与新成本未准入。[具体协议](metadata-batch-frame-protocol.md)。 |
| 只按B/图桶缓存返回对象或metadata mini graph | 已停止该原样方案 | 缺seq/max lengths、actual/causal/partition、物理表和副作用；V1 FULL并不捕获这些producer。携带完整动态字段与更新协议的graph方案仍另列待确认。 |
| 旧residual cohort的device-only metadata/DMA小方案 | 已停止同一独立1%局部预算 | 112周期全native prefix免费最高仅0.81838%本地周期；保留host时低值。此判断不剪较大host联合parent、其它负载或宣称C8通用上界。 |

### 有成本的情景估计

[复算](metadata-host-net-price.json)保留两个global/local parent builder、每组return_replace、first producer/copy与store。只假设：plan lookup与qsl提交的25/26可合并，local-common和query-view工作有f比例可共享。按同代配对计算，f=0.25/0.5/0.75时组件中位毛预算为1.68886/2.08714/2.49014ms。实际可移除且落在关键路径的比例ρ和新增非重叠成本h都**尚未测量**。

`净省 = ρ × 组件毛预算 − h`

| f | ρ | h(ms) | 局部条件净省(ms/迭代) | C8整批条件改善 | C8 P90条件改善 |
|---|---:|---:|---:|---:|---:|
| 0.5 | 0 | 0.25 | -0.25000 | -0.3053% | -0.0838% |
| 0.5 | 0.5 | 0.25 | 0.79357 | 0.9690% | 0.2660% |
| 0.5 | 1 | 0.25 | 1.83714 | 2.2432% | 0.6157% |

这些是诊断B4成本与原C8请求包络组合的**有利情景**，不是matched实测净收益。rho=0的回退没有裁成0收益。原C8有227个real waves，按截止时间和完全串行传播敏感性，整批1%需净0.81898ms/wave，P90需2.98371ms/wave。f=.5/h=.25时rho约需≥0.51217才到整批1%参考。只认每请求首输出一次时，短slot7需4.48846ms、P90需14.91855ms；不能将“已开始但尚不构成首输出依赖”的异步下一波自动累加为TTFT收益。

六普通请求按实际1+verification次数参考，跨两轮response参照，1%需净约0.6277–2.0793ms/迭代；这是成本目标，不是把B4时间当B1实测。1%仍只是历史排序参考，不是用户验收条件。

**下一步**：沿上述BatchFrame每个observer和V1 metadata release追到真正末读，决定能否做具体grouped producer；不要重跑旧CPU-plan或仅测设备小核。DP packet留下明确序列/净成本缺口继续主图；随后回末层attention/FFN row-demand producer和剩余输入/embedding/Engram交付。

本轮新NPU0、模型加载0、服务推理0、新profile0、远端写入0。仅只读171冻结源码及旧DB schema、在本项目重放旧renderer与已有profile/request账；未更改服务或其它环境。
