# F08 短长组：同 scope 其他任务的相对顺序

## 结论与范围

复用单次历史采集中的 640 个 T=8180 选中三元组，对每个选中 scatter 的设备区间，与**同 layer scope 内经 host flow 唯一匹配且有 kernel CSV 行的其他设备任务**做区间求交；其他 track 的交集先求并集，再除以 scatter 区间。使用 Decimal；不是两次独立实验。短、长的阈值分别为事后选择的 <50 µs 与 >200 µs。所有 640 scope 的 sidecar thread 与单一 trace scope 一致，scatter flow 各唯一匹配，开始时间、task ID、时长与原选中记录核对一致。采集经过固定提取器挑选，见 `diagnostic-20261029/extract_target_layer_all_kernels.py`；没有 CSV 行的同步事件等不在本交叠统计之中。其余 scope、其他线程的任务、未被采到的设备活动也不在范围内。

| 选中范围 | 短任务 | 长任务 |
| --- | --- | --- |
| 全部 640 | 31 条都没有同 scope、其他 track 的**已匹配**交叠任务 | 609 条都有，交叠并集占 scatter 区间 89.84%–100%，中位数 93.29% |
| 同 EP3、同 db852 摘要 40 条 | 29 条均无上述交叠 | 11 条均有，交叠占 92.82%–100%，中位数 93.17% |

所有 609 条长任务，均与同 scope 某条 `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3` 记录区间相交；其中 1 条另外还与一条 `alltoallAicpuKernel` 区间相交。31 条短任务也都有 3 或 4 条该名称的 MatMul 匹配记录；最近的一条全在 scatter 结束之后，间隔 10.96–37.56 µs，中位 22.8 µs。同 EP3 子集短组间隔 10.96–37.56 µs，中位 22.3 µs。因此，短组缺少交叠不能简单解释为该 scope 没有此类 MatMul 被匹配。按现有选择，两组同一 track 内未观察到其他已匹配任务与 scatter 相交。

上表是**同一个 profiler 导出的区间关系**：事件重叠不证明同时占用相同硬件资源，也不证明一个任务使另一个变慢；scope 内未见交叠不表示该时段设备空闲。同一类名称可能对应多个 MatMul，最近者的选择是时间上的，无张量身份和依赖保证。两类各自 scatter 的时长决定交叠机会，本结果不能独立于时长作为随机对照或因果证据。

## 可比性与未决因素

全部 640 条名称相同，CSV `Model ID` 都为 `4294967295`；此数值的语义未由同版采集器文档或实现绑定，不能判定 eager、graph replay 或排队阶段。EP3 同摘要子集还共享 scatter 物理 stream 35、设备 track 35、host tid 2128，同一导出形状、类型和 block 见 `F08-SCATTER-VARIATION-AUDIT.md`。这些排除了在**这些已导出字段上**不同，但没有保证两组实际 slot 内容、张量地址、block 有效工作量、缓存状态相同，也没有证明采集完整或同一硬件状态。

本脚本独立复算：全部短组与长组 scatter duration 中位数 17.7 / 258.935 µs，所列 `aiv_total_cycles` 中位数 1,207,815 / 1,197,499；同 EP3/db852 的 29 / 11 条分别为 17.7 / 261.08 µs 和 1,207,815 / 1,249,383。这是**两组分别取中位数**，不是逐项配对，cycles 口径及其与墙钟的对应尚未证明，不能换算成真实忙碌时间或等待时长。EP3 既有原 DB / trace 的两个样本核验及口径边界见 `F08-SCATTER-VARIATION-AUDIT.md`。

下一步可在已有原始 trace 中针对 EP3 选定短、长样本检索全设备 track 的时间交集、事件关联与同步信息，先查记录完整性，再查是否有共同的数据或资源关系；如果缺少接口或 PMU 采样定义，则保留时序分组与争用、共同阶段、事件配对和导出误差等并列假说。

## 复算

在现有任务目录执行 `python3 agents/fusion/f08_scope_overlap_audit.py`，只读取已保存 JSONL、triples、variation；输出 `f08-scope-overlap-audit.json`（含 EP3 逐层聚合）。输入 JSONL SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`，脚本 SHA-256 `16058045c493beabcc871e955dc8eaae71f15ff8c87090f5780bb24306a6ba94`，输出 SHA-256 `3b4b4356d96fc57c1f61ed619948616091254cc479fb470bb0f30af8ae1277d1`。未启动 NPU 负载，未修改目标实现。

## 同类现象扩展到 T=4760 的 8 卡、40 层

上文分析固定 T=8180；此处使用相同的两份冻结聚合输入和同一类型、仅限该 scope 已关联任务的区间规则，选择 T=4760 的 **320 scope：EP0–7 各 40 层、同一摘要 `7ddf7ae9…`**。每 scope 唯一关联选定 `RmsNorm→InplacePartialRotaryMul→ScatterNdUpdateSk` 三个 flow，三者同 stream、task ID 连续；这提供同次 scope 的前后记录，但不等于已证明 tensor/storage 身份。三者导出的类型、输入输出 shape、dtype、format、核心类型、block 数在 320 个 scope 中完全一致。

| 同摘要同层配对 | 40 层选中 scatter 时长 min / median / max |
|---|---|
| EP3 | 14.540 / 16.130 / 17.460 µs |
| EP0 | 146.81725 / 152.207 / 160.277 µs |

各层在各卡独立选中并取本卡时间差，没有跨卡时间戳相减。同层同摘要和物理签名相同，排除了将**已导出的这些字段不相同**作为两组差异的解释；不保证有效 slot、真实写入量、地址/stride、缓存、tiling 分支、资源状态或请求身份相同，不能归因于某张卡。

按事后阈值 `<50 µs` 和 `>100 µs`，短 41 条（EP3 40、EP2 1），长 279 条（EP2 39、其余 7 卡各 40）。区间检查仅针对每个 scatter **同一个已选 layer scope** 中，已有 host flow 关联且有 kernel CSV 行、同 device track PID 而不同 track TID 的其他任务；已核验 scope 内 flow 唯一与 device PID 一致。这不覆盖 scope 外、没有匹配 CSV 的事件、未导出或其他进程设备活动。短组 41/41 没有与这类其他 track 任务相交；长组 279/279 均与该 scope 的 `QuantBatchMatmulV3` 命名任务有正区间交叠。短组每 scope 也有 3 或 4 条该命名任务，但未与 scatter 相交，故不能仅解释为短组没有匹配到这种任务。相交不证明资源争用、依赖或性能原因；不相交也不证明整卡空闲。

该结果将上文 T=8180 的相关性观察扩展到另一个形状，并排除“所见只存在于 8180 行所选样本”这一狭义说法；两组来自同一已保存 profiler 采集，不是独立重复实验，不可合并当作基线请求次数或收益。前述 T=8180 的事后长阈值为 `>200 µs`，此处因不同任务时长选择 `>100 µs`，阈值及组的差别不等于硬件类别。复算 `python3 agents/fusion/f08_4760_matched_rank_overlap.py`；脚本 SHA-256 `5342a3cad4c0c76fe6d16a9e4480ef6d561620d2973163b7890cff6b5e90625b`，逐条结果 [f08-4760-rank-overlap.json](f08-4760-rank-overlap.json) SHA-256 `ea892e7faf87ad87e010b3c40c5a4f5a5ecbc711ec06b7d2e9107b3e9cbcab5f`；CPU 读取已有文件，未操作 NPU。

## T=4760：短 scatter 与前三事件跨度

对上述 320 个 scope，沿同一已选 stream 计算 norm 结束→RoPE 开始、RoPE 结束→scatter 开始的间隔，并以 norm 开始到 scatter 结束的两个端点计算跨度。**跨度包括三个任务以及中间两段间隔，不是三个算子时长求和。** 从同卡同 scope 的导出时间相减，未跨卡相减。

| 事后分组 | norm→RoPE 间隔 min / median / max (µs) | RoPE→scatter 间隔 min / median / max (µs) | 三事件跨度 min / median / max (µs) |
|---|---|---|---|
| 短 scatter，41 条 | 0.020 / 582.840 / 1952.900 | 186.740 / 704.820 / 1953.340 | 242.240 / 1233.880 / 3962.080 |
| 长 scatter，279 条 | 0.0195 / 0.0205 / 0.04075 | 0.01925 / 0.02025 / 0.0405 | 189.09625 / 197.576 / 213.676 |

该已选样本内，短 scatter 对应的三事件跨度全部长于长组，因此不能把“scatter 单条记录更短”直接解读成这三个已选端点之间的跨度也更短。这是 profile 局部事件关系，不能推断全层、整个请求或关键路径性能；间隔不是设备空闲或等待时间。上文 T=8180 的类似相关性来自同一次采样，扩展到 T=4760 并非独立重测。

同一摘要 L9 的两个例子：EP0 三条设备 task 为 13778/13779/13780，scatter 149.637 µs、RoPE→scatter 间隔 0.02025 µs、三事件跨度 194.076 µs；EP3 task 为 13718/13719/13720，scatter 14.560 µs、间隔 1201.120 µs、跨度 2576.160 µs。两卡各自在已存 SQLite 内按设备起点与 device ID 唯一核对 TASK 和 `COMPUTE_TASK_INFO`，每个任务各有唯一同 connectionId 的 CANN `launch`，并通过 `CONNECTION_IDS` 分别关联唯一的相应 PyTorch 提交 API。CANN 的三条 launch 在各自同一个线程，PyTorch 的三个提交 API 在各自另一个线程。RoPE→scatter 的 **CANN launch 起点差**在 EP0/EP3 分别为 935.270/1210.850 µs；**PyTorch 提交起点差**分别为 1560.710/1687.960 µs。两种差值各自只在同一表、同一卡、同一线程比较；它们属于不同记录，不与设备间隔相减，不代表设备排队、端到端调度或 causality。EP0 和 EP3 的 trace 与 DB 同源，不是独立计时实验。

这些核查不能确定两组的 slot 数值、有效工作量、实际 stride、tiling key、运行二进制或竞争关系；不能认定更长间隔由下发或任何特定因素造成。需要逐次捕获 slot/有效写入信息、实际 kernel/tiling 标识和同口径设备活动或同步对象后，才有可能继续区分上述假说。

本地复算 `python3 agents/fusion/f08_4760_gap_spans.py`，脚本 SHA-256 `9ec9d28534feeb9fd5cfb258550e032b408186bcd6bf836f18057e0ae7fc04f3`，结果 [f08-4760-gap-spans.json](f08-4760-gap-spans.json) SHA-256 `382bf6809fd005b8140eba6a4b9f45260ce7b246d3dadfdb67ee70c78ea742ac`。设备 task 与提交关系复算 `python3 agents/fusion/f08_4760_ep0_ep3_db_binding.py`，校验本任务 stage 身份和两份 DB 哈希，脚本 SHA-256 `31c7584af6bfa88e3f5695217af88a43c18488fd10a6da6ae16da7055fba31a6`，结果 [f08-4760-ep0-ep3-db-binding.json](f08-4760-ep0-ep3-db-binding.json) SHA-256 `88121d339ef3847eced126c2f03055dc7edfc9597af354062eca551af0cba400`。只读既有数据，未运行 NPU 或修改实现。

## T=4760 的 L9 两任务：已导出的索引、tiling 与计数范围

限定上节 EP0 task 13780 / globalTaskId 5198（长）与 EP3 task 13720 / globalTaskId 5004（短），分别读取对应卡的同次 SQLite 和 trace。两个任务 `COMPUTE_TASK_INFO` 中除 globalTaskId 以外的已导出 17 字段及字符串解析结果完全一致：逻辑算子为 ScatterNdUpdateSk、`blockNum=48`、`opState=dynamic`、`attrInfo` 为空，shape/dtype/format 与上节一致。按 device track PID、stream 和 task ID 在各自 trace 中各唯一选到一条 X，核对其起点和时长与各自 SQLite TASK 一致；X 的 args 只有 Model Id、Task Type、Physic Stream Id、Task Id、Batch Id、Subtask Id、connection_id，没有展示索引数值、有效 slot 数、真实 stride、binary 或 tiling key。

每任务的 `TASK_PMU_INFO` 各有 28 个同一时间戳的字段，名称集合相同；两条各自导出的 `aiv_total_cycles` 为 **1001709 / 1004199**，而 TASK/trace 时长分别为 **149.637 / 14.560 µs**。这排除了“本次两条导出的 AIV cycles 数值相差约十倍”的说法，不能由此得出两次实际工作量相等、硬件忙碌时间相等或多出的时间都是等待：这些是同采集链的导出计数，历史 exporter 的归一化、单位及与墙钟的对应尚未绑定，同一次 PMU 时间戳也不是 task 内每段的运行采样。

两个例子的已匹配导出字段中没有可逐项核算的 slot 内容、有效/重复坐标计数、实际 cache stride、tiling key 或执行二进制身份；空 `attrInfo` 和未见对应字段都**不代表这些值为空或为零**。早期 scheduler descriptor 的摘要 `7ddf7ae9…` 对应总 Q=4760，可参见 [调度关联记录](F12-SELECTED-SCHEDULER-BINDING.md)，但 scheduler 数值和物理 shape 不证明实际 slot 内容、是否有效或与设备 tensor 的身份。这次有界检查不能排除不同有效写入量、实际布局/tiling 分支、硬件资源状态或 profiler 时序口径这些解释，也不能选定其一为根因；最小补证仍是同次 task 绑定的脱敏 slot 聚合、runtime kernel/tiling 和 stride，再结合可靠的同步与时间计数。

复算 `python3 agents/fusion/f08_4760_pair_metadata_audit.py`：只读两卡指定旧 DB 和对应 trace，脚本核验两种输入各自 SHA、task/stream 唯一性和设备时间一致、导出字段比较；脚本 SHA-256 `8f8585cd1701a6d4c2fea32716c9e98650f2445bb0998f37c82eef9537ee4d17`，结果 [f08-4760-pair-metadata-audit.json](f08-4760-pair-metadata-audit.json) SHA-256 `9429a6f0c64b4ff1fa5d0db1015fa68c863274e04bfdeb13fe46d4611f6d5117`。仅限上述数据库与两个精确 X 事件，不声称已解码全部 raw profiler slice；无设备操作或目标代码修改。
