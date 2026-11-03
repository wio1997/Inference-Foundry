# F08：EP3 L9/L30 同轨、事件与下发记录

针对 `F08-SCATTER-GAP-STRATIFICATION.md` 的两个既有样本，限定同一次原始 EP3 trace 和 DB；容器内以只读方式查询，未启动 NPU。时间只在同一导出表或明确指定的同一设备轨内比较；该导出并非独立重测。

| 选中样本 | 同一设备 track / stream 的 rotary 结束→scatter 开始 | TASK 与同轨 trace | 同一 globalTid 的 CANN rotary launch 结束→scatter launch 开始 |
| --- | ---: | --- | ---: |
| L9 短写入，task 13225→13226 | 1693.280 μs | 仅两端 | 1687.370 μs |
| L30 长写入，task 13467→13468 | 0.020 μs | 仅两端 | 432.160 μs |

trace 设备 PID `2179523`、track 35：以上窗口内只见两端 X 及与两端时间吻合的 flow 终点。DB 的 `globalPid=2128, deviceId=3, streamId=35` 窗口中也只见这两条 TASK。因此，可排除这个特定查询口径里存在一个已记录的中间同轨 TASK；不能排除未采集内容、未以 TASK 记录的同步、其他线程、图重放、其他设备轨或其他进程。`torch_to_npu` flow 和 DB connectionId 的语义不可混为一谈。

扩大到该 trace 同一设备 PID 的其他 track 的 X，L9 两端间没有 X；L30 的 0.020 μs 窗口与 track 47 一条量化 MatMul 的导出区间相交。**这只是指定 PID 已导出的 X 的筛选，不能推出整卡空闲、同轨之外没有硬件工作、或重合导致任务变慢。** 未校验跨轨时钟及采集完整性，不将重合用于精确因果分析。

在同轨两端外扩各 300 μs 的已导出 X 中，L9 的 EVENT RECORD 在 scatter 后，L30 的 EVENT RECORD / WAIT 在 rotary 前，另外 EVENT RECORD 在 scatter 后。尚未绑定 event 对象及配对关系，不证明它们控制了这两个任务，亦不能因间隔内未见 EVENT WAIT 就排除同步等待。

四个选中 TASK 的 connectionId 各唯一关联到 CANN_API 的一个 `launch`，这些 launch 属于同一 globalTid。对每组这两个 launch 之间完整包含的 CANN_API 行进行同 globalTid 查询，两组均只查到两个 `aclrtGetStreamAttribute` 和一个 `ScatterNdUpdateSk_Tiling`。后者的记录持续 0.670 μs（L9）与 0.680 μs（L30）；可以排除**这条已记录事件自身持续时间**单独构成毫秒量级的下发间隔，不能排除其他线程、未采集调用、Python 工作、其他 tiling 或调用间资源等待。调用可能嵌套，不能把数量相加当耗时。主机 launch 间隔也不能跨轨与设备时间相减解释为延迟，不能证明它是 scatter 时长变化的原因。

## 复算与后续

四个脚本均在当前任务的 stage 容器运行，读取既有数据，不读取请求内容：

- `f08_ep3_gap_sync_audit.py` → `f08-ep3-gap-sync-audit.json`：trace 固定设备 track 与 DB TASK 精确范围。
- `f08_ep3_gap_events_audit.py` → `f08-ep3-gap-events-audit.json`：同设备 PID 其他 track X 及同轨附近 event。
- `f08_ep3_gap_flow_audit.py` → `f08-ep3-gap-flow-audit.json`：TASK 与 CANN/PyTorch API 的 connectionId 精确关联。
- `f08_ep3_host_launch_interval.py` → `f08-ep3-host-launch-interval.json`：同 globalTid 的两次 launch 间完整包含 API 聚合。

脚本和结果 SHA-256 按上述顺序分别为 `49f6e0c53134df179925ba7ce49f7035a5f94161c4b1359e5d438a54d12cee29` / `7e44de5fb6d1b47f5343da2467c7c9ede9ad1f8c9e432f66b4de8c9a57adbf09`、`0e3f2c2e4fd43e819265185432ce160450a6e81b392b5d7564fe08002d6b63fc` / `ac6cf41e12c0f92fa41c71fe4238350ccb57564a3c4cf04f6b9b70cacfbe2ae9`、`b864dbebea4a0291b50c267b82f7750ed2aab91c6744469adaaa483a8eed3d9e` / `783540d37d503db37660db50e6e4f18f6626a422f15df549ea73247c63f092d6`、`ef97602d3963de5c59faec4ae144e21d7d4d0d1e94be1ae0ddb68dae27602a80` / `6ab4fc57dd84a211c6a16b375cffcdfdbdcc2ba321d23414de40f58eb29f583a`。

后续可先查同版采集与事件导出覆盖，再追同线程 PyTorch、CPU 栈和 event 对象关联；如果无法从已有 trace 绑定，不推断因果。仍需具体 slot 内容、流之间资源活动以及 profiler/PMU 时间语义才能解释两种 scatter 耗时。
