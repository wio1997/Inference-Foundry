# EP8 第 0 层 MC2：其他 stream 的时间重叠

只检查一个 rank（EP8）、三个已归属 target 第 0 层内部采样窗口中的 `DispatchFFNCombine`，按同一 rank 的 `kernel_details.csv` 时间戳，把其余数字 stream 的导出任务裁剪到 MC2 区间并求并集；每轮选中的 MC2 均为 model 49、stream 89。结果不是跨 rank 比较，也不是端到端请求耗时。

| 样本轮次 | MC2 时长 | 其他数字 stream 已导出区间并集 | MC2 内无其他数字 stream 导出行覆盖的时间 | 其他 stream 命中 |
|---|---:|---:|---:|---|
| 1 | 2035.220 μs | 0 μs | 2035.220 μs | 无 |
| 2 | 944.321 μs | 296.554 μs | 647.767 μs | stream 92，`allgatherAicpuKernel`，1 行 |
| 3 | 1661.606 μs | 357.112 μs | 1304.494 μs | stream 92，`allgatherAicpuKernel`，1 行 |

轮次 2/3 的区间重叠分别约占各自 MC2 时长的 31.4% 和 21.5%；两条其他 stream 行的 model ID 为 `4294967295`，不能仅凭时间相交把它们归属到该 MC2、该请求或该 target 层。三次均未找到类型名称以 `hcom_` 开头的其他数字 stream 行，也未找到 stream 非数字的重叠行；**这不表示 MC2 没有通信**：该内核自身已包含通信和计算，CSV 不提供其内部阶段划分。

“无其他 stream 导出行覆盖”仅是已导出 CSV 的集合差，不表示整卡或其他 stream 实际空闲，更不证明 MC2 在纯计算或纯等待。`allgatherAicpuKernel` 的记录名与时间相交不单独证明真实链路传输、设备利用、资源冲突、隐藏效果或可节省耗时。没有逐请求／张量或通信 ID 的因果关联；仅检查 EP8 的三个样本，不能推广到 16 rank、其他模型阶段或基准服务。

复现脚本为 `analyze_ep8_mc2_other_stream.py`，只读容器包装器为 `run_ep8_mc2_other_stream.py`，结果及其源文件和窗口 SHA 保存在 `ep8-mc2-other-stream.json`。窗口和脚本 SHA 已与本地文件复核，EP8 kernel CSV 哈希与 `evidence/TRACE-EXPORT-SHA256.json` 相符。ZCode 完整 stage 名为 `operator-ceiling169-v2-097-ep8-other-stream`，模型验证为 `deepseek-flash`，对应 `.exit-code` 为 0。`no_device_work=true` 指**此查询没有启动新的设备任务**，不是对原有时间区间内设备是否有工作的判断。
