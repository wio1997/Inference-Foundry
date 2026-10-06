# Challenger：H1 关闭范围与 H2 phase packing — 2026-10-06

**建议 PIVOT：关闭 H1 当前实现路线；KEEP H2 的一次设备区分诊断；STOP 拆批实现与大型 E2E。** H2 是值得检查的结构性可能，并非已证实的最大可消除 Gap。Current=None，没有新增代码级 PERF_KEEP。本次只读本地源码/raw，并写本文件；未启动现场操作。

## 亲读与独立核验

沿用此前 review.md 的规则、源码身份核验；本轮重读 H2_REVIEW_PACKAGE、H1_DECISION，并亲读 native_scheduler 的 running eligibility 与 waiting admission、native_async_scheduler 的 PP cadence、native_pp_utils 的 slot 消费、ascend_patch_spec_pp 的 accepted+draft 合并发布、native_model_runner 的 empty execute、EngineCore 与 lazy Future。额外读 commit_trace.py:181–206，确认 reply.begin 位于原 enqueue_output 之前，published_upper_bound 位于返回之后，pending metadata 随原异步 FIFO 映射。

亲读 Run242/sol_raw/core.jsonl、rank8.jsonl、rank0.jsonl 的 batch18/19/20 原始记录，并独立遍历全部记录归约；没有使用 sol_correlation 的结论代替重算。三个本地文件 SHA 分别为 134bd5578a90dd3dd969c035f34c267bbf20bb2d20c6d8c8ef15460e0f6aabc8、8edcc1668737c35689d330495b991e355926d411de157c8c8ae18e12de3627f1、c160fc5acd2cbbf0ba428020b89e1fd144e38f3e2534cc9410912f9e7e29647e；前者与 rank0 身份直接与 trace_summary 所列核对，rank8 本地 hash 也保留供追溯。

从原始 commit.output 按 request 前缀对齐四个 request*.events.json 的 token_ids：四条各 64 tokens，与 commit 中 token 序列逐一完全相同，均有 DONE。此为诊断轨迹一致性证据，不替代完整 correctness 或产品 SLO。

## H1：支持关闭，但限定命题

独立按同一次 core step 中“额外 schedule.begin→旧 commit.output”配对，再关联 rank8 同 batch 的 execute/sample reply：得到 **28 个有效输出旧提交全部 producer 尚未进入原 enqueue_output；28 个已提前发布旧提交全部为空输出**。有效输出的 producer.begin 比该 schedule.begin 晚至少 10.525483ms，最大 1555.787888ms（包含启动阶段）；不能把这些差值当优化空间。必要 sample reply 尚未由唯一原生产者发布，是有效的不可获取下界，避免了“晚观察到 ready，所以之前不 ready”的逻辑错误。

具体 raw：core 行249–254 已安排并 enqueue batch19 的四个请求；行258 才提交空 batch18。rank8 行250–251 已提前发布 batch18；故提前 drain 该空结果不会使已经 enqueue 的 batch19 更早获得 eligibility，也不会产生 SSE token。core 行263–266 enqueue batch20 空步时，batch19 sample reply producer 直到 rank8 行271 才进入；行275 才标记发布返回；core 行272 为该有效提交。

**可关闭的命题：** 在 Run242 的原生 reply/FIFO 语义、该短提示四请求轨迹上，没有观察到 enqueue 优先延后“当时已经可获取的有效结果”；H1 不能支撑 drain-first 性能 patch。

**不能扩大为：** 所有负载、长 prefill、其他队列深度/并发、结构化输出或其他 epoch 下永无 H1；也不能推出 worker 中更早的 copy-ready 不存在、无法重构结果生产、更晚输出链没有成本。后者是不同假说，当前没有理由激活。28 个相邻样本是同一短轨迹的相关样本，不是 28 次独立负载验证。

小措辞修正：最后一个提前发布空提交 batch56 后是 batch57 空清理，不宜笼统称“所有28个空提交后均已enqueue非空Decode”；这不改变 H1 关闭，因为该尾部也没有有效 token 可提前提交。

## H2：有结构依据，净收益仍 unknown

源码事实：native_scheduler.py:440 每次调用增加 current_step，504 限制 next_decode_eligible_step；native_async_scheduler.py 把一次已排请求的下一 Decode 设置为 current_step+PP；running 与 waiting admission 没有显式均衡 PP phase 的约束。native_model_runner.py:1219–1228 在 empty return 前仍消费 PP slot。Ascend 补丁把 accepted 与 draft 在 propose 后同包广播，设备侧消费遵守同一 FIFO 与 generation。

原始轨迹事实：batch1 只有1请求，batch3–53 的奇数 batch 为4请求，batch55 为2请求；偶数 batch 为空，batch57 为尾部空清理。新请求在 batch3 与已有请求同相，源码允许其后维持同相。

推断：独立请求分到不同 PP phase 可能使另一个 cohort 在某 stage 等待上游/下游时提供可用工作。但 **“每隔一个 scheduler slot 空”不是“stage 闲一半”**。零 token 步可能快速推进状态，先前 target/draft/communication 同时仍在设备上执行；copy_event host 等待的 50–70ms 也可能主要是必要计算。

更强反例：4→2+2 可能重复模型权重读取、增加 Graph/collective/dispatch 次数并降低算子批量效率；即使 stage idle 减少，净 tokens/s、TTFT/TPOT 仍可恶化。PP stage 不均衡及末 stage MTP 的串行工作也可能决定周期下界。当前没有半批成本、stage overlap、可执行队列或设备时间，无法证明 phase balancing 值得实现。暂不选择另一架构。

## Gap 排名与 Top 3 代码问题

按下一诊断价值排序，**不是已测可消除时间排序**：

1. H2：同相请求是否让设备 stage 出现可被独立 cohort 覆盖的空档？有代码及 host 轨迹依据，大小与可消除性 unknown。
2. 同一依赖链上的 target/MTP/PP通信是否已经占据关键路径，使 host copy wait 主要为必要工作？这是 H2 的竞争解释，随同一诊断区分，不开第二路线。
3. H1 已ready有效结果晚commit：本样本不支持，退出 active；全产品上界及其他 workload 仍 unknown。

对应最高价值代码问题：

1. admission/running 选择在哪一步把 cohort 归到同 phase；是否能在不破坏公平性、长 prefill、preemption/KV/grammar 条件下形成独立 cohort？先查可行约束，不实现策略。
2. 当目标 stage 真正闲置时，另一个请求 cohort 的 token/draft/KV/PP slot 是否已经可用，且不会因 stream/collective 排序使其仍不可执行？“有请求”不等于“有合法工作”。
3. 若设备证据阳性，填空收益能否覆盖拆批增加的必要工作与 stage 不均衡？固定 cap/resources/MTP 的单一候选才有归因价值，最终由完整 matched E2E 裁决。

## 最小区分诊断

同意一次短、有界、固定条件的现役原生 4×64 streaming 设备采集作为下一步；不是批准性能 patch。拟用已核验 dynamic profiler 路径、当前容器命名空间 PID、≤15s/all16 workers，不由我执行。采集必须实际覆盖四请求活跃窗口及数个稳态 Decode 周期；仅有启动/收尾轨迹无区分力。记录实际 queue cap/resources 与当前 epoch，不能从 installed.queue_capacity=3 推断运行 cap：该 marker 在首次 cap wrapper 生效前；主 Agent 报告 native log 的 SELECTED cap2/resources3，本轮未亲读该远端 log。

最小验收是实际可解析的设备 compute/communication/stream 事件、rank→PP/TP 映射、时间覆盖、时钟关联和工作量 ID。须区分 target、draft、PP payload/HCCL 与 D2H；不能仅有 API start/stop 200、进程列表、host-only event 或空 profiler 目录。PROFILING_MODE=dynamic 导致前轮 API 无设备产物的原因本轮由包/主 Agent 报告；我没有把缺失设备数据补成任何设备结论。

决策：

- 可重复 stage 空档与 cohort 同相对齐，并有合法独立工作可覆盖，且成本上界支持非微小收益：H2 获得候选价值，才设计一个 native-control patch 与 correctness；仍需 matched 完整 E2E。
- stage 已充分重叠，或主要受必要 target/draft/通信约束，或拆批成本明显抵消：STOP H2，依据实际 profile 重新选最大 Gap。
- 仅观察到空 slot/host wait，或缺 rank、时间/算子归属、只有 idle 无可用独立工作：INCONCLUSIVE。不要自动升级到拆批实验或大型 E2E，也不反复盲修 profiler。

一个未改策略的 profile 最多支持必要条件和收益上界，不能直接证明 phase-balanced 反事实的净收益；若需要半批成本，须在阳性后明确一次可改变决策的验证，不做 batch-size sweep。

## Stop List 与误读提醒

停止 H1 drain-first/done guard；停止以空 slot 比例计算设备利用率或预报2倍收益；停止修改 PP eligibility/FIFO 来删空步；停止 cap/budget/cadence/MTP/拓扑扫描；停止把 target/draft host span 相加当周期；停止在设备数据无效时启动大型 E2E 或新 Runtime 重构。

最可能误读的证据是“4请求同phase+28空步=可消除stage bubble”。现有 raw 只证明调度序列，不能证明设备空档及净可消除工作。第二是将 H1 的 producer 下界扩大成设备 token 未ready的结论。第三是把28次相关观测、4×64 DONE 或 profiler HTTP成功当通用性能/功能证明。

今天重新开始，我会关闭已被该轨迹否定的 H1 实现，做一次真正设备可见的 H2 诊断，并保留否定 H2 的明确出口；不提前承诺 phase balancing，也不将本次路线评审记为性能进展。
