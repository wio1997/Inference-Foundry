# F08：EP3 线程状态证据与采集覆盖边界

限定同一份 EP3 rank 3 历史采集、L9 和 L30 样本，同一主机工作线程 `globalTid=9139690412736`（拆为进程 2128、线程 6848）。此前两个 launch 间的大部分落在 rotary 的 Dequeue 和 CANN API 作用域内：这是调用区间的覆盖，不是该线程的 CPU 时间。另一条线程的 Python 调用栈和关联 Enqueue 也不是该工作线程的运行时栈。详细关联见 `F08-EP3-HOST-CPU-INTERVAL.md`。

## 此次找到的采集配置

同份 `profiler_info_3.json` 记录 torch_npu 2.10.0.post7、CANN 9.2.0、CPU/NPU 活动、`with_stack=true`，以及 `experimental_config._host_sys=[]`。同版本容器内 `torch_npu/profiler/experimental_config.py` 的 `HostSystem` 定义列出了 CPU、OSRT 等选项；这只是配置项含义的源码证据，不保证当前磁盘安装的代码二进制就是历史进程所加载的版本。

原始 profiler 的 `host/sample.json` 更具体：`host_cpu_profiling=off`、`host_osrt_profiling=off`、`hostAllPidCpuProfiling=off`、`sys_profiling=off`，同时 `hostProfiling=true` 且 `taskTrace=on`。所以不能把“host profiling 已开启”解读为 CPU 线程运行/阻塞状态已采集：API 记录存在，但对应 CPU、OS 运行时采集项配置为关闭。记录的是配置，无法仅从它保证采集程序的每项实际执行细节。

原始主库 `ascend_pytorch_profiler_3.db` 的表包含 `PYTORCH_API`、`PYTORCH_CALLCHAINS`、`CANN_API`、`CONNECTION_IDS`、`TASK`，没有以 OS 调度/线程状态命名的独立表；`analysis.db` 和 `communication_analyzer.db` 的表仅列出通信分析与 `StepTraceTime`。这是数据库表清单，不是对原始二进制切片或其他未列文件的穷尽解析。已有目标工作线程的 Dequeue 记录未附 callchain；现有 Python 栈来自另一 producer 线程上包含 Enqueue 的调用，不能作为该工作线程等待/运行状态的栈。

## 导出完整性信号

同采集日志在 02:43 报告 TraceStepTime、Communication 和 KernelView 无法获取 step range，step node list 为空；底层导出日志还提示 step_trace 数据库连接失败以及频率数据无法获取。这些告警限制以 step 分组和实际频率推断，不能由此断言所选 API/TASK 记录均无效、事件一定丢失、或分析中的每条数据均不完整。历史样本的两端任务与调用关联应按先前逐条证据使用，不能补上不存在的 step 对齐或实际频率。

在本次检查的 `logs/*.log` 和 `mindstudio_profiler_log/*.log` 中，筛选了上述几类告警并保留精确来源；这不是所有可能丢记录方式的审计，未确认是否存在数据丢失或其他未记录系统活动。

## 诊断结论

现有证据不能将约 1.69 ms / 0.43 ms 的 API 作用域拆成 CPU 实际执行、系统调度延迟、锁/运行时等待或 profiler 引入的额外耗时；也不能把未观察到的内部调用当作没有发生。已配置的 CPU/OSRT 采集项关闭，并且目标 Dequeue 缺该线程 callchain，解释了为什么单靠已导出的调用表无法分解这两段时间，但并不证明等待或 profiler 开销就是原因。

要区分这些假说，需在同版本、相同负载下获得能绑定目标工作线程、覆盖对应区间的系统调度/线程状态和用户态栈（以及采集开销对照），并校验采集有效性与时间映射；如果要追具体阻塞还需可解析的等待对象或同步事件。当前目标禁止新 NPU 负载，本次没有启动新采集；这些是将来可区分假说所需的证据，不是现有数据可以补出的结论。

复算脚本 `f08_ep3_capture_coverage.py` 在现有 stage 容器只读历史配置、只读 SQLite 表清单和指定告警；结果 `f08-ep3-capture-coverage.json` 保存文件摘要、表名及警告。未读取请求、模型权重或修改推理实现。
