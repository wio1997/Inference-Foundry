# Prefix-cache 与退役交付判断

Goal ACTIVE。本文记录一段有限的源码/配置/已有事件判断，不表示 prefix 或全链路已经穷尽。每个替换原语逐位一致；本轮无模型运行、native 准入或实测加速。

## 介绍

沿 request block hash → 本地 prefix 查询 → 多组命中协调 → touch/分配/发布 → worker block table/实际 cache 读写 → finished/abort/preempt/reset/复用追踪。先看整段是否已有共享、是否确有清零/拷贝成本，再判断能否合并局部操作。与已有 metadata-host、物理 cache 和跨请求 selected-KV 候选重叠的范围不重复计价。

## 到了哪里

- 重新以统一入口核对171主机，当前原容器镜像仍匹配。27个运行树源文件中19个匹配先前本地件、8个新取；这里只是所列文件身份，未声称 loaded bytecode/全部二进制/实时health。逐段范围见 [source review](prefix-retirement-source-review.json)。
- 保存的 serve.sh SHA 仍为 `c2972011…`，没有 KV/EC transfer、generic CPU offload/prefetch 或 KV offloading 参数。当前默认源与构造 guard 绑定这些分支为空；独立已完成 profile 启动记录又确认 `enable_prefix_caching=True`、async scheduling、block128。sparse KV offload 默认False，DSpark V1亦不满足其支持前提；Engram 的 `cpu_offload=True, dp_shared_memory=True` 是另一条活路径，不能随之排除。
- Request 仅为新完整 block 延长 chained hash，父 prefix、token、LoRA/MM/embedding/salt extra identity 都属于 key；不是每步重算整个 prefix。group ID 又是池查询 key 的一部分。Hybrid lookup 已按相同spec/manager/pool分组，full-attention命中在定点迭代中可直接裁剪，不能再次计成新共享。
- lookup 为 logits 限制到 `num_tokens-1`，不等于最后一 token 可以无条件消除。DSpark `use_eagle()` 为True，仍保留 trailing drop。固定布局无 Mamba、cacheable MLA/SWA 的 block128，非cacheable circular state32；命中维持 scheduler LCM128 对齐，部分块 CoW 条件不成立。circular state 每请求私有，不作为跨请求prefix命中内容。
- 新块的 GPU zeroer 不能从 `needs_kv_cache_zeroing` 单一标志推出：实际 `NPUWorker.initialize_from_config` 还要求 V2 或 Mamba＋Eagle3。固定 V1＋DSpark 失败此 guard；`_zero_block_ids` 通过 `hasattr` 成为无 zeroer 的空GPU调用，DSV4.1 MLA/SWA也不是普通C8-MXFP scale-cache分支。
- 全16已完成 trace 的 SHA/size 均与上一段一致，专属cache zero/copy标签和 `prepare input` 标签均0。此录制不能独立证明 Python 函数未调用，也没有录到整个 engine scheduler/lookup parent。现有 generic zero/fill 的 CPU/enqueue/dequeue/native多层总量只是导航，不能归给cache清零或相加算净省。

## 结果

| 范围 | 状态 | 判断与价格边界 |
|---|---|---|
| 融合/删掉“现有每次分配的通用KV block清零”GPU工作 | 已停止 | 当前worker guard下未初始化此zeroer，原GPU工作0，可省GPU及其请求信用0；新增kernel不能节约该不存在的工作。CPU新block-ID收集、传递、drain仍存在，属于更大host交付，未因此定价为0。普通aten zero/fill、C2 ring写入和启动tensor zeros不在此排除范围。 |
| 融合“现有部分prefix-hit CoW stack＋copy” | 已停止 | 当前对齐、无Mamba及无remote-tail路径下不产生该拷贝，可省GPU/请求信用0。新的细粒度命中/布局机制会改变条件，需重新判断；不推导出一般无需CoW。 |
| 删已有remote KV/EC、sparse offload及通用权重prefetch搬运 | 已停止 | 在不变冻结服务中未启用这些路径，可省现有搬运0。引入新offload/owner/layout以换取容量是另一候选；不否定该方向，也不排除活跃Engram CPU表读取。 |
| 仅翻cascade/trailing-drop开关，或跨原M几何复用cached hidden | 已停止 | async speculative配置主动禁用cascade；波动draft尾不能忽略。通用两段softmax或不同M/tile中间值复用不继承逐位一致，已有changed-M反例仍适用。原几何shared-read重写另保留。 |
| typed physical-cache ownership＋不可变BatchFrame＋退役协议 | 待确认 | cache/host大交付值得保留；但没有整个engine scheduling/lookup/ID bookkeeping的同负载CPU父区间，不能套旧worker边界。缺物理owner/代次、所有最后读者、pin/event/storage/backpressure新成本；与既有metadata-host及global-empty快路不相加。 |
| 跨请求shared-prefix/selected-KV加载共享，保留原逐row算术 | 待确认 | 可改变融合边界，但prefix命中已省去的历史计算不能再算。缺实际各rank选中物理行交集、同代owner、现有L2共享、活Q集合与原softmax/reduction几何；六普通负载另需匹配群体和时间。它与已有SMLA/QLI unique-row/物理cache范围重叠，不是新实测收益。 |
| 本轮新准入 | 已确认可实现：无新增 | review/源码反例通过不能当作数值、native或模型性能通过。 |

## 退役边界对联合实现的约束

原 `BlockPool.free_blocks/get_cached_block/is_block_writable` 在惰性fixture上执行：ref_cnt由1减到0、块进入free queue后，同一prefix仍可命中，且该块仍不可写。这是原方法的控制语义反例，非真实池状态dump。真正新分配才撤销hash并通知reuse watchers；reset在仍有非null引用时失败。不能把finished、CPU free或ref_cnt0直接解释为可擦除、可与另一层同址覆盖或所有device读者已退役。

命中引用先全部touch，再进行可能驱逐其它组的外部分配。发布hash只覆盖finalized/committed token；async placeholders、in-flight量、拒绝回滚和SWA保留窗口分开维护。只以乐观num_computed_tokens提前释放窗口，会误伤前轮仍读的块。

本服务无KV consumer connector，源码 `defer_block_free=False`；不能声称当前已有通用seq fence。原路径依赖worker原device提交顺序，而新异步搬运、提前输入、跨层驻留/megakernel必须显式建立最后读者与storage/range lease。带KV consumer的延迟free、pending push和failed recv源码条件已追到guard，但当前未启用，不能凭其实现给新机制背书。

worker在无scheduled tokens返回前仍执行 `_update_states`，消费finished tombstone、移除request/input_batch及侧cache。其代码明确区分同ID的旧finished与新request；这不等于已证明 ingress/旧future 的全局ID代次安全。下一段继续 normalized internal request ID、EOS/stop/length normalization、abort旧future、异常上抛与恢复的实际调用者。未闭合的startup warmup/MM/helper内核也仍记录，不能把依赖名字具名当成扫描完成。

## 证据

[主判断](prefix-retirement-frontier.json)、[当前源身份](prefix-retirement-execution-identity.json)、[依赖](prefix-retirement-extra-dependencies.json)、[白名单配置记录](prefix-retirement-config-record.json)、本地 `prefix-retirement-profile.json` 全16只读提取、`tools/analyze_prefix_retirement_frontier.py`。原方法fixture只证明所述key/引用控制关系；注入digest serializer的hash反例不冒充实际vLLM key字节或碰撞分析。

NPU0、模型加载0、服务请求0、新profile0、远端写入0、服务变更0。读取均为自有目录下的既有文件与原容器源，未操作另一个profile控制器。
