# 169 EngineCore 框架侧采样入口：有界核查

## 更新：独立 py-spy 已准备（2026-10-30）

在主控授权后，stage 容器 DNS 查询失败，采用一次 Mac 官方下载与分块转送。PyPI 官方 [py-spy 0.4.1 元数据](https://pypi.org/pypi/py-spy/0.4.1/json) 中 aarch64 wheel SHA-256 为 `ee776b9d512a011d1ad3907ed53ae32ce2f3d9ff3e1782236554e22103b5c084`；解包的独立 ELF SHA-256 为 `972918b5d6af92649ec086ffdd89cefa0d725fd14844434c766f58494a1c4f8e`。ELF 保存在本 run 的 `tools/py-spy-0.4.1-aarch64`；因 baseline 没有挂载 run 分析目录，同哈希副本放在本人 runtime 的 `tools/py-spy-0.4.1-aarch64-972918b5` 供容器执行。没有 pip 安装或修改基础镜像。

baseline 容器中 `--version` 显示 0.4.1，`record --help` 和 `dump --help` 均可运行；[官方发布记录](https://github.com/benfred/py-spy/releases) 显示 0.4.0 已加入 Python 3.12 支持。核对 baseline ID、NPU 进程归属及双 engine 空闲后，仅对 DP0 PID 699 做过一次 `dump --nonblocking`，未启用 `--locals`，退出码 0。结果保存在本人 runtime 的 `logs/pyspy-idle-dump-900cd35f.json`；这是空闲附加可用性的证据，不是负载期性能数据，也不能替 DP1 证明采样一定成功。

现已准备 `enginecore_framework_controller.py`：保留完整 frozen warmup、measure1 seed 和 8 请求验证；正式请求时对两个 EngineCore 各执行 `py-spy record --nonblocking --idle --threads --rate 50 --duration 20 --format speedscope`，并并发 procfs CPU 计数。仅本人容器，独立目录记录采样器启动/结束时间、stderr、逐请求时间与每 engine 指标；没有执行该控制器或新负载。两种采样都可能引入扰动，`--idle` 的栈样本比例不是 CPU 时间，非阻塞读取可能产生残缺栈。先由主控审阅；在同次结果中还须区分业务时间段及尾部空闲，确认对应的 engine 主线程后才可讨论函数分布。

以下是安装前的检查记录，关于工具缺失的结论仅适用于当时状态，已由本节更新。

结论：**当前未发现可直接附加现有服务并取得 Python 调度函数栈的已安装工具。** 可使用已存在的 Linux `/proc` 接口对两个 EngineCore 做一次最长 20 秒的低频 CPU 计数采样，但这只能回答“进程和主线程有多少 CPU 运行时间、采样时主线程是什么状态”，不能替代调度函数级 trace，也不能把主线程等待位置解释成调度瓶颈。两个备选脚本已准备，尚未执行。

## 本次只读检查

- 169 用户 baseline 容器中，`shutil.which` 未找到 `py-spy`、`perf`、`gdb`、`pstack`；另在 `/usr/local/bin`、`/usr/bin`、`/usr/local/python3.12.13/bin`、`/opt`、`/vllm-workspace` 检索未发现 `py-spy`。这限定为上述路径的检查，不声称整个主机绝无工具。本次没有安装、拷贝或执行采样器。
- 当时目标容器内存在 `VLLM::EngineCore_DP0` 和 `VLLM::EngineCore_DP1`，PID 分别为 699/720；PID 随服务重启变化，脚本会重新按进程名识别并检查 starttime，不硬编码 PID。
- 同一容器中执行只读检查的进程及两个 EngineCore 当时 UID 0、`CapEff=0000003fffffffff`、`Seccomp=0`，`/proc/sys/kernel/yama/ptrace_scope=0`。其中包含 SYS_PTRACE 能力，但这些条件**不证明**当前有 `py-spy`、目标进程内存可被读取，或未来 attach 一定成功；此次没有尝试 attach。
- 实际容器的 `ms_service_profiler` Python distribution 记录 26.0.0；`/usr/local/Ascend/ascend-toolkit/latest/share/info/msserviceprofiler/version.info` 是组件安装记录 26.2.0。这两项不等同 CANN `profiler_info.cann_version=9.2.0`。已安装 CLI 的 `cli.py:30–42` 展示 analyze、parse、split、compare；未发现其中有可附加已运行进程的命令。检查 baseline 的 PID 1 与两个 EngineCore 环境键，未见 `SERVICE_PROF_CONFIG_PATH` 或 `PROFILING_SYMBOLS_PATH`。
- [vLLM-Ascend Service Profiling Guide，main](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html) 的 MS Service Profiler 方法要求服务启动前配置路径；修改配置开关只对已按该方法准备的服务有意义，符号配置变更还需重启。该在线 main 指南不是本容器软件版本的完全等价证明；基于已见入口和环境，不能承诺对当前服务直接启用。此结论不排除未查到的其他方法。
- [py-spy 官方说明](https://github.com/benfred/py-spy) 说明它支持对进程 PID 采样；Linux attach 受权限限制，`--nonblocking` 可避免暂停目标但可能产生不完整/错误栈。此处缺少可用二进制，故没有可直接执行的 py-spy 方案，也没有做权限探测式 attach。

## 已准备的最低成本替代

`sample_engine_core_proc.py` 在容器内按准确命令名找到两个 engine-core，采集总长 20 秒、目标间隔 0.2 秒：

1. `/proc/PID/stat` 的进程级 `utime`、`stime`、进程 state 和 starttime；以及 `/proc/PID/task/PID/stat` 的主线程同类字段。
2. 主线程 `wchan` 与 `schedstat` 原文。缺失、无权限或读失败时记录错误而非造值；识别到重启/PID 复用则终止并保留不完整记录。
3. 每条记录含 monotonic 和 wall-clock 时间及 `SC_CLK_TCK`。结果存入容器的 `/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/logs/enginecore-proc-<唯一标识>.json`，该可写挂载的宿主路径是 `/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/runtime/logs/`；文件使用独占创建，不覆盖旧文件。

`launch_engine_core_proc.py` 是本机提交入口，先校验原 baseline 容器 ID，再在该容器中运行上述脚本。**本次未运行**。主控若决定在已有真实工作负载期间采集，可在本地运行：

```bash
python3 /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-analysis/launch_engine_core_proc.py --execute
```

采样本身不会发起请求或 NPU 任务；如果服务空闲，只能得到空闲期结果。应使用已有真实负载的时间窗，并在分析时关联同次客户端/引擎指标；不要为了该脚本擅自启动额外负载。需要量化影响时以同负载的无采样数据对照。

## 解释边界

- 根据 [Linux `proc_pid_stat(5)`](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)，`utime/stime` 以 clock ticks 表示：两点差除以 `SC_CLK_TCK` 是对应区间 CPU 运行秒数。进程级计数涉及该进程线程，主线程需单独看 `task/PID/stat`；它们不包含其他 worker 进程，进程聚合 CPU 秒数除以墙钟秒数可能超过 1。
- [`wchan`](https://man7.org/linux/man-pages/man5/proc_pid_wchan.5.html) 是**采样时主线程**的内核休眠位置，不是累计等待时长，也不说明本次请求、通信、设备或 Python 函数的原因。`schedstat` 原文是该主线程统计，不等于进程所有线程的等待，更不能直接分解 vLLM 调度链。符号受权限/内核配置限制。
- 采样时间间隔、clock tick 粒度和线程瞬时变化会隐藏短工作；20 秒记录不能证明整个 1K 输出的逐步骤情况。还需核对实际工作负载、各 DP 分配以及机器/容器时钟对应。
- **未知：** Python scheduler 每个函数的 CPU 耗时、排队原因与逐请求关键路径。进程 CPU 计数不能给出这些结果；如必须取函数级轨迹，需要另有现成可用的外部 Python 采样器，或后续使用预先配置的隔离诊断服务。当前不安装工具、不重启 baseline，不做猜测性调参或另一次设备采集。
