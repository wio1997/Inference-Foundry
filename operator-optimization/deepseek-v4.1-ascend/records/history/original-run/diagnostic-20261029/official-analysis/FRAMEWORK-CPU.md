# 169 EngineCore：一次真实负载的 CPU 框架采样

**结论：两个 EngineCore 主线程在请求窗口内各消耗约一个 CPU 核；全采样窗口的主线程栈经常位于等待 worker 回复的路径。** 这为研究共享内存回复队列和轮询提供了一个具体入口，但尚不能证明它限制了端到端吞吐，也不能把等待路径的栈占比换算为纯调度开销、通信时间或可节省时间。

## 场景和采集

本机 ZCode 对 `enginecore_framework_controller.py --execute` 恰好执行一次，run ID `framework-20261030T091949Z-3d9bcf87`，退出码 0；回执为 `framework-cpu-zcode-receipt.json`。在本人原 baseline 上先完成 frozen warmup，再 seed measure1，随后进行 8 路 streaming 请求。每个请求为 32768 输入、1024 输出 token；前缀缓存查询 262144、命中 235904（约 89.99%），成功 8、抢占 0。两个 DP 各 4 请求；measure1 输入 hash 与冻结值一致。

两个 EngineCore 各自使用独立 `py-spy 0.4.1 record --nonblocking --idle --threads --rate 50 --duration 20 --format speedscope`，并并行采 `/proc`；均退出码 0，stderr 为空。启动和结束时 owner 记录均属于本人 baseline，结束时两 engine running/waiting 均为 0，health 检查成功；未重启服务或重置缓存。本次采样对目标进程有扰动，不是无采样的吞吐基准。

原始文件位于 169 任务目录对应的 `runtime/logs/framework-20261030T091949Z-3d9bcf87/`，包括两份 speedscope、`enginecore-proc.json`、计数和时序。`framework-cpu.json` 是 CPU-only 匿名聚合与原始文件哈希；`framework-cpu-public.json` 是不含 prompt、生成文本和 PID 的公开摘要。在本机运行同目录的 `python3 run_framework_cpu_analysis.py` 可通过 SSH 在 stage 容器只读原始文件、重算 `framework-cpu.json`，不触发设备工作。此复算依赖 169 和原始文件仍可访问；公开摘要本身不能独立重建全部样本。

## 业务窗口内的 CPU 计数

8 个请求从最早发起到最晚结束为 **15.600 秒**。procfs 每约 0.2 秒采样；只选处于这个窗口内部的 78 个点，两端相减覆盖 **15.434 秒**，而非整个 15.600 秒；计数粒度为 100 ticks/秒。

| 对象 | DP0 CPU 时间 | DP1 CPU 时间 | 说明 |
|---|---:|---:|---|
| EngineCore 进程全部线程 | 15.19 秒 | 15.18 秒 | 分别约 0.984、0.984 个 CPU 核的平均运行量。 |
| EngineCore 主线程 | 15.12 秒 | 15.12 秒 | 分别约 0.980、0.980 个 CPU 核的平均运行量。 |

这是一段真实请求期间的 Linux CPU 运行计数，不是 NPU 利用率、Python 函数 CPU 分解或单个请求的计数。主线程 78 个瞬时状态样本中两边均有 76 个为 `R`、2 个为 `S`；状态和 `wchan` 都只是采样时刻，不能由此推算阻塞总时长。

## 栈样本：主线程与辅助线程分开

py-spy 目标为 20 秒，**显式包含 idle**；进程实际从启动到退出约 DP0 21.54 秒、DP1 19.98 秒。每个主线程分别得到 **876 / 881** 个样本，名义权重合计 **17.52 / 17.62 秒**。样本数量低于以 50 Hz × 20 秒计算的名义 1000 个；输出没有每次采样的绝对时刻，不能假设固定 20 ms 间隔并精确裁切业务开始和结束。请求已结束后的尾段可能进入样本，栈比例不等同于业务窗口占比。stderr 为空且没有空栈，但仍不能排除非阻塞读取产生的局部不完整或漏采。

| 样本口径 | DP0 | DP1 | 限制 |
|---|---:|---:|---|
| 主线程栈包含 `multiproc_executor._wait_for_response` | 70.89% | 77.75% | 全采样窗口内按主线程样本权重计算；包含 idle，不能解释为 CPU 耗时或已暴露等待。 |
| 主线程栈末端为 `shm_broadcast.wait` | 32.08% | 32.58% | 包含该函数不同代码行；不能全部称为自旋或阻塞。 |
| 主线程栈末端为 `threading.wait` | 27.74% | 21.45% | 可能包含请求前后 idle；不与上一行或含父栈行相加推导关键路径。 |
| 主线程栈末端为 `sched_yield` | 8.79% | 9.88% | 只是该 Python 栈位置的抽样权重，不是系统调用 CPU 时长。 |

四类辅助线程单独看：两个进程的输入线程样本几乎都在 ZMQ `poll`；输出线程大多在 `queue`/`threading.wait`；signal-callback 在 `threading.wait`；worker-monitor 在 `select`。这些是按**各线程自身样本**得到的描述，包含空闲和等待，不把多个线程百分比相加当作 CPU 使用率。

## 源码关联与下一步判断

在两份保存的主线程栈中，可以直接看到这条父子调用链：

`step_with_batch_queue:706 → future.result:98 → _wait_for_response:103 → get_response:433 → dequeue → acquire_read → wait:204`。

按同一全采样窗口的 inclusive 栈权重，`_wait_for_response` 出现在 DP0/DP1 分别 12.42/13.70 的名义秒权重中；其中带 `step_with_batch_queue` 父栈的分别为 12.10/13.02，另有 `execute_dummy_batch` 父栈分别 0.32/0.68。它们是相互包含的栈统计，不是可相加的阶段耗时，也无法从样本精确裁去尾部 idle。

运行镜像的 `vllm/v1/engine/core.py:206–216,235–238,630–719` 已实现 batch queue：允许时先 schedule、新 batch 非阻塞提交，队列未满且仍有可调度工作时可先返回；队列满或无更多可调度工作时取最早 future 的 `result()`，之后处理结果并更新 scheduler。`vllm/v1/executor/multiproc_executor.py:91–103,426–433` 中该 future 通过 `get_response` 等待 message queue 的 `dequeue`。因此采样明确关联到**已有异步队列取结果**，不是证明 schedule 函数本身计算缓慢，也不能把该时间解释成通信传输耗时。

冻结启动参数含 `--async-scheduling`；服务日志报告 `pipeline_parallel_size=1`，冻结源码 `vllm/config/vllm.py:589–599` 对该配置推导 `max_concurrent_batches=2`，这是根据日志、启动参数和源码得到的值，非读取当前进程中该字段的独立快照。调用栈实际出现 `step_with_batch_queue`，也证明样本中该路径被执行，但不证明每个时刻队列都满或始终存在可并行的后续批次。

`vllm/distributed/device_communicators/shm_broadcast.py:191–206,770–794` 描述读端检查共享缓冲，并在特定 busy 条件下调用 `sched_yield`，否则可进入 socket poll。**尚不知该次运行的 busy-loop 实际参数及其各阶段持续时间，也没有对应事件证明它位于全请求关键路径。** CPU 计数和栈采样可以共同提示该路径值得调查，但它们的时间范围不完全相同，不能将主线程 CPU 全部归于轮询或判定轮询已造成延迟。

一个具体的结构候选是：在保持 batch queue 顺序、输出可见性和请求依赖的前提下，研究将回复就绪等待交给通知驱动的机制或专门等待执行体，让 EngineCore 只在确有独立批次可调度时继续工作。需要同时核对 GIL/线程唤醒、共享队列语义、可并行批次数和设备完成时点；如果后续工作依赖尚未返回的 token，等待无法被消掉。当前没有证据证明该改法降低 TTFT/TPOT 或提高吞吐，不能把 CPU 用量或栈比例当作预计收益。

更有针对性的后续分析应先对齐同次 worker 执行、回复 ready、engine 读取和请求阶段，再区分实际可消除的 host 开销与等待设备结果的必要时间。当前结果不要求重复负载，也不构成更改实现或估算提速的证据。
