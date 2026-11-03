# 169 重采样：scheduler / runner 元数据只读复核

检查范围是当前新诊断已有 `metadata-summary-early.json`、16 worker 与其他进程的 `events-*.jsonl` 中白名单字段及客户端合成请求 ID；只通过本人 stage 容器读取，未启动设备任务、读取请求正文或更改服务。该分析是 profile 窗口的关联线索，不代表整个请求或全流程。

## 摘要复核和请求关联

早期摘要 SHA-256 `2e322b98e2f972e35d74cb2d5fc98dd7071a0ff48f38f8b47a62d91a2b48538a`。独立按文件重算：23/23 个事件文件 SHA 均与摘要一致；453 条成功 scheduler 输出和 192 条 runner 入口的 canonical schedule digest 均一致，未发现摘要计算错误。24 个 digest 各对应 **1 条 scheduler 和 8 条 runner**；另有 426 个仅 scheduler 的 digest 组。scheduler 的 `PROFILE_ACTIVE` 门控可能长于 worker 的 profiler 启停区间，所以仅 scheduler 不能视作设备未执行。16 个 worker 每个有 12 条 runner 入口，只说明捕获窗口，不等于全程。

按指定固定 key 重新对 `cmpl-<客户端合成 request_id>-0` 做 HMAC，8 条客户端合成 ID 与 scheduler/runner 匹配均为 **0/8**；复核直接给定 ID 和响应 ID 的有限几种固定形式也均为 0。23 个 `instrumentation_loaded` 事件均记录 key-present=true，这个布尔值不能证明每个进程 key 完全相同；scheduler 与 runner 的已见 request HMAC 各有 8 个不同值，两边集合相同。这支持调度与 worker 之间存在一致的匿名标识，但**不证明它们与客户端哪一条响应一一对应**。主控另在当前环境源码中查到 engine 将 external ID 后拼随机后缀；本次摘要本身不含后缀或映射，不能通过枚举恢复，也不能认定其他输出一定没有映射。

旧 client 的 `expected_engine_request_hmac_sha256` 因 client 环境读取到空 key 而不可用，复核没有使用它。早期摘要的 8 个 synthetic-index 条目全零表示**不匹配、缺少关联**，不是没有调度、prompt 工作量为零或请求失败。`new_computed_tokens` / `cached_computed_tokens` 是输出描述结构的字段名，不是“本轮新计算 token”或实际 prefix hit 数；`num_computed_tokens_post_schedule` 为调度更新后的状态，不是更新前值。`spec_token_count=null` 是该 hook 未记录可数 list/tuple，不能当作 0。

## 推荐的具体设备核查对象

选取摘要 `db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8`：scheduler `pid=690, call_id=690-2` 与 8 条 runner 入口（`736-5, 753-5, 1113-5, 2128-5, 2165-5, 2711-5, 3556-5, 3612-5`）的规范化摘要一致；scheduled_total=8180，3 个匿名行的字段如下：

| 输出结构 | computed 字段 | scheduled | 调度后 computed | 调度后 prompt |
| --- | ---: | ---: | ---: | ---: |
| new | 29568 | 3200 | 32768 | 32768 |
| new | 29440 | 3328 | 32768 | 32768 |
| new | 29568 | 1652 | 31220 | 32768 |

三行分别满足 `computed + scheduled = post computed` 且区间落在 prompt 长度内，适合优先查 8 个对应 worker 的 `wio_diag:NPUModelRunner.execute_model:<call_id>` trace 标记及其内部设备任务（仍需匹配同一文件/进程与时间域）。在**当前冻结 scheduler 源码字段语义和实例确实匹配**、该字段表示本次连续调度起点的条件下，这些是计划的 prompt 区间；它们不单独证明完成的 token、唯一客户端身份、设备耗时、prefix 命中量或关键路径。三行 spec 均为 null，不能简单地按 hook 结果断言无 speculative 值。

另一个摘要组 scheduler `705-1` 同为 total 8180，三个 new 行 computed 都为29440，scheduled 为3328、3328、1524，匹配另外8个 runner。两个组分属不同 scheduler，不能仅凭序号或时间认定它们同时、对应同一次跨 DP 请求或同步运行。

全部 scheduler 输出中有 8 行 new descriptor：computed=29568 为3行、29440为5行，scheduled 加总22888；8行 post 均等于 computed+scheduled，6行到32768、2行未到，且全部 spec 为 null。这是被门控记录中的描述符统计，不是已完成 prefill 的全量计数、去重后总量或硬件计算量。其余1766行有 cached descriptor；cached descriptor 也不等于缓存命中，未逐条视为实算。

## 方法局限

摘要 `summarize_metadata.py` SHA `bcd13ec15b53245d3b10922ed2e2caab19e26b244bf783d064a52fe10fc50a00`：只在摘要合法时统计HMAC，使用匿名 synthetic index；无法关联时每请求输出的零须与覆盖数一起解释。digest 基于排好序的请求 HMAC 和调度字段，内容相同可以跨多步重复；1→8只说明当前文件中该摘要的一对多匹配，不自动证明全局唯一 replay。检测到 0 条 instrumentation_error 和额外重复 call_id 不排除未插入、未记录或工具内部缺失。不能用主机和设备绝对时间直接相减而不核实时间域；也不能把未导出事件当成无工作。

已保存 `agents/bounds/metadata-path-snapshot/vllm_scheduler.py` 的构建与 `_update_after_schedule` 顺序与上述前后字段解释相容，但与本次运行二进制/源码版本仍须独立绑定；动态 metadata、worker trace 与匿名调度的关联，须先核对具体 marker 是否保存在该 worker 的 trace 和正确回放中，再研究实际算子。不需要重新采集才能进行这项后续只读核查。

## 两个 8180 token 调度组：草稿槽预算的条件性解释

冻结启动命令 `evidence/frozen-serve.sh`（SHA-256 `c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba`）请求 `max-num-batched-tokens=8192`、DSpark 和 `num_speculative_tokens=5`。保存的 `agents/bounds/draft-constraints-snapshot/spec_config.py`（SHA-256 `9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19`）在第 1461–1462 行对 DSpark 设置 `parallel_drafting=True`；其第 1853–1891 行定义的新增 drafting 槽数，在 `method=dspark`、`K=5`、不是 draft_model 的条件下为 `K−1=4`。保存在 `agents/bounds/metadata-path-snapshot/vllm_scheduler.py`（SHA-256 `c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884`）第 581–584、1035、1080、1262–1263 行的路径，每次安排新请求时先从剩余 input budget 为该请求保留这些额外槽，再扣除该次安排数和保留槽。

如果这两个调度组实际采用上述源码和配置、初始 input budget 为 8192、没有其他工作先消耗预算、两个完整 prompt 段先被调度、另外的 token budget 及长 prefill 阈值等没有更紧约束，则第三个请求被截断，三行具体数值**均与已保存调度描述符相符**：

| 已保存组 | 三个剩余 prompt 长度，假定处理次序 | 条件模型安排量 | 合计 |
| --- | --- | --- | ---: |
| `db852bff…` | 3200，3328，3200 | 3200，3328，1652 | 8180 |
| `1b826883…` | 3328，3328，3328 | 3328，3328，1524 | 8180 |

计算为每轮对输入预算依次取 `min(请求剩余, 剩余预算−4)`，再扣该值与 4；两组均恰好满足 `8192−3×4=8180`。例如第一组前两行共安排 6528 并保留 8，第三行最多安排 `8192−6528−8−4=1652`。这是源码与两组数据**一致的候选机制**，不是对实际截断原因的证明：摘要不保证原始处理顺序，历史 resolved `parallel_drafting` / `max_num_new_slots_for_drafting`、源码版本和各 budget 的动态值未直接绑定，其他限制也可能得到相同数值。它也不证明 12 个额外槽是实际执行的硬件工作、显存填充或可消除的耗时；两组位于启用了 profiler 和 hook 的历史诊断中，不能推广到无 profiler 基线。

无设备复算：

```sh
python3 - <<'PY'
for lengths in ((3200, 3328, 3200), (3328, 3328, 3328)):
    budget, scheduled = 8192, []
    for length in lengths:
        take = min(length, budget - 4)
        scheduled.append(take)
        budget -= take + 4
    print(lengths, scheduled, sum(scheduled), budget)
PY
```

## 同一 scheduler 两次调用的匿名请求推进

原始 `runtime/metadata/events-690.jsonl` 在本人 stage 容器内只读核对 SHA-256 为 `26e4e81afbd719f8509fc0a45dc61d5600a9f7a8620d9a8a9c20b057bdd254ec`，与早期摘要相同。文件第 5 行是 `690-2` 的 call_end、digest `db852bff…`、total 8180；第 7 行是 `690-3` 的 call_end、digest `7ddf7ae9…`、total 4760。二者同 PID、调用号连续、文件顺序和 hook `time_ns` 顺序一致，结束点相隔 8.53534 ms。**这只是主机 hook 两端点的间隔，不是设备执行耗时或请求延迟。**

本机 `diagnostic-20261029/capture-summary.json` SHA-256 `d3862f7c3ff92628aaefd4d39bda62352fc557a4f6c4fc2a48c24ca8318746e8` 的两个条目中，第一批 3 个匿名键全都出现在第二批 4 个键中；两批的 request 和 post-state 各自键非空且唯一。按相同键连接（表中仅以数字描述匿名行，不输出标识）：

| 第一批 new computed | 第一批 scheduled | 第一批 post computed | 第二批 cached computed | 第二批 scheduled | 第二批 post computed | 第二批 spec count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 29568 | 1652 | 31220 | 31220 | 1548 | 32768 | null |
| 29568 | 3200 | 32768 | 32768 | 6 | 32774 | 5 |
| 29440 | 3328 | 32768 | 32768 | 6 | 32774 | 5 |

因此，同一调度器内匿名键和状态变化支持这三个请求从第一次已记录的调度状态延续到下一次；其中部分 prefill 行的 `31220→32768` 与 `1652+1548` 精确衔接。`sitecustomize.py` 的同进程 key 在初始化时读取且以 HMAC 对请求字符串计算标识，样本 `requests` 按 HMAC 排序，**排序后的行序不是原始调度次序**。身份与字段语义仍以已保存 hook/源码适用于这次运行、不发生标识碰撞为条件；post 是 scheduler 更新后的计划状态，不能单凭它证明设备完成、关联客户端响应或测量 prefix hit。第二批这两行 `spec count=5` 仅表示 hook 从那次输出读取了长度为 5 的 list/tuple；不能把第一批的 null 当成 0。

这条关联**增强了分段调度的解释**，但不直接给出第一批请求处理顺序、动态输入预算、其他限额或实际配置，不能将 `8192−3×4=8180` 升格为已证明的截断根因；也不证明预留槽产生了设备计算或性能损失。第二批的 4760 包含一条当次 new 行的 3200、上述 cached 行的 1548 与两条各 6，不可把 4760 整体归给一个请求。
