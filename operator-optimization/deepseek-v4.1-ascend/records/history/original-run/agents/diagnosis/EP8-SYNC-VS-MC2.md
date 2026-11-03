# EP8 目标层中同步事件相对 MC2 的位置

**范围：**复用 `ep8-l14-l39-flow-intervals.json`，即同一次 EP8 trace 中，L14 的 `737-20` 和 L39 的 `737-45`。这些 X 通过同层 host scope 的已选 flow 唯一关联；选择条件和全局排除项见 `EP8-L14-L39-FLOW-INTERVALS-20261029.md`。这不是所有同步事件、所有设备流或整轮负载的覆盖。只比较各自与同一 trace 中对应 MC2 的时间，没有跨卡时钟比较。

|层|选中且去重的 EVENT WAIT|选中且去重的 EVENT RECORD|与对应 MC2 相交|MC2 后|
|---|---:|---:|---:|---:|
|14|16|21|0|0|
|39|12|17|0|0|

这些事件均在对应 MC2 开始前结束。同范围内按原提取规则的 X 身份 `(track pid, track tid, 相对 MC2 起点的时间)` 去重，没有一条 X 被多个选中 flow 重复引用。此结果不代表未选 flow 和其他 rank 也有相同结论，也不证明等待依赖与 MC2 的因果关系。

另一个值得保留的现象：L14 关联的 track 35 / task 13268 WAIT 持续 32.276 ms，结束于 MC2 开始前 6.817 ms；track 39 / task 26722 WAIT 持续 29.079 ms，结束于开始前 7.234 ms。二者区间相互重叠，不能把时长相加。L39 的 track 35 / task 13566 WAIT 为 12.517 ms，结束于 MC2 前 3.943 ms。这些事件属于不同 track；没有事件依赖 ID、缺失流或任务内部阶段的证据，无法断定它们等待同一对象、阻塞 MC2 或解释 L14 的约 160 ms MC2 内核区间。Profiler 扰动也使这些时间不宜外推至无 profiler 基线。

在任务根目录运行 `python3 agents/diagnosis/summarize_ep8_sync_vs_mc2.py > agents/diagnosis/ep8-sync-vs-mc2.json` 可复算，不读取原始大 trace、不访问 NPU。结果含输入 SHA、逐层唯一 X 数、位置分类和最长 WAIT。缺口是其余 rank、同层未被选择的 flow、未记录活动及事件依赖；后续若无现成可绑定数据，不应把同流/邻近时间当作依赖证明。
