# EP15 固定层范围的外部同步与 MC2

本次只读复用同一次受 profiler 扰动的采集，不是线上性能或因果归因。EP15（PID 3602）、scheduler digest `1b826883…`、L14 scope `3602-20` 与 L39 scope `3602-45` 的 MC2 锚点分别是 task 29168、30364；锚点引用自 `diagnostic-20261029/target-layer-mc2-16rank.json` 并在提取时按任务 ID 与起始时间重新核对。按照同线程 host 层 scope → `torch_to_npu` 唯一 start/finish → 唯一设备 X 的规则进行筛选；仅比较每层对应的同一 trace 的相对时间，不跨 rank 相减。

| 层 | 全部选中 flow | 唯一 X | EVENT WAIT（均在 MC2 前结束） | EVENT RECORD（均在 MC2 前结束） | hcom X（均在 MC2 前结束） | MC2 时长 μs |
|---|---:|---:|---:|---:|---:|---:|
| L14 | 104 | 104 | 16 | 21 | 3 | 160382.47575 |
| L39 | 70 | 70 | 12 | 17 | 3 | 8231.7155 |

两个 scope 的选中 flow 均唯一匹配 X，未匹配或模糊均为 0，反向多 flow 对同一 X 的额外引用数为 0。该结论只覆盖所选 flow，而非完整的设备活动；筛选前全 trace 有非唯一 start/finish 的 flow 被排除，本次该全局计数为 0。时间按半开区间判断。三个 hcom 名称依次为 `allGather`、`alltoall`、`reduceScatter`；此名字与提交范围并不能证明它们的依赖、实际网络传输或请求关键路径。

L14 最长选中 WAIT 在 track 35 为 32287.435 μs，结束于 MC2 前 6811.103 μs；另一 track 39 WAIT 为 28974.56125 μs，结束于 MC2 前 7326.83275 μs，区间有重叠，不能相加。L39 最长 WAIT 在 track 35 为 12510.89 μs，结束于 MC2 前 3945.661 μs。没有 event record/wait 的关联 ID，不能断言它们等待谁，也不能排除未选中或未记录活动、间接影响与 MC2 内部等待。仅能排除这些已绑定 X 作为与对应 MC2 显示区间直接相交的事件；不能用它们解释 MC2 异常时长。EP8 的对应结论见 `EP8-SYNC-VS-MC2.md`，二者都是样本，不代表全部 EP8–15。Profiler 扰动下的时长不适用于无 profiler 基线。

## 证据与复算

- 固定 EP15 trace `dp1_pp0_tp7_dcp0_ep15_rank7_3602_20261029023525647_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`，SHA256 `788ed028080baf043abf55638f18b56da0716dd0105a026cb7c144a5cce84fe4`；先前保存于 `diagnostic-20261029/scope-summary-v2.json`，提取前按 SHA 验证。
- `ep15-l14-l39-flow-intervals.json` SHA256 `5ea71228d72a30379233ffe195807de424d4e76cd9cfda7785765e95be6eaf96`；含所有选中 X、相对时间和 flow 引用。
- `ep15_l14_l39_flow_intervals.py` SHA256 `4eb895f08254edd60a1dd3dffd9e33e87649cad7ac513e6e3eb9586094e3734d`；`run_ep15_l14_l39_flow_intervals.py` SHA256 `6f8e91f874f216b7bbe72ad41a16489c3223f9e96cc62deeeeb3ad2208fbcf7d`。复算为 `python3 agents/diagnosis/run_ep15_l14_l39_flow_intervals.py`，其前后检查自有 stage 容器 ID、镜像及挂载，在该容器内以 nice 15、ionice idle 读取，不操作 NPU。无需重复运行即可读取 JSON。
