# Engram 视图与 worker 排队：低扰动诊断的可行边界

本文件仅是方案评估。没有启动负载、没有修改目标推理代码；是否可准确观测 worker 的 enqueue→dequeue，当前仍是 **UNKNOWN**，不能承诺单靠这个方案测出或解决排队。

## 现有无 profiler 日志能否回答

只读检查本人 baseline 和 selective2 的两份固定 service.log：分别约 227 KB、271 KB，读取前后 stat 一致；两者均有 72 行包含 `engram`，但限定词 `stride`、`storage_offset`、`contiguous`、`enqueue`、`dequeue`、`inplacecopy`、`slice_tiling` 均 0 行。只输出行计数和 SHA，不输出日志内容；这些是指定文件和拼写的限定搜索，不排除其他命名或其他日志。已有 sidecar `_tensor_metadata` 只记录形状、dtype、device type，不记录 stride、offset、storage identity、queue ID 或异步消费完成时间。当前无法从这些日志还原实际 view stride、storage 生命周期及 worker 排队。

上一份旧的 stack/shape 关闭 profiler 显示若干同名调用的短时 Slice_Tiling，但没有 shape/stack，不可与新采样逐次配对，也不是无 profiler 样本；详见 `EP8-ENGRAM-LOW-STACK-COMPARISON.md`。

## 第一部分：只观察视图，不改变源码

已安装 `AscendParallelEngramEmbedding.embed_gathered(self, gathered, num_tokens)` 定义于 `engram/embedding.py:333`；源码第 342 行会对传入 gathered 调用 `contiguous()`，lookup 要求内层 unit stride。建议以后在**独立诊断容器**中通过外置、可关闭的 wrapper 仅在该方法入口观察：

- PID/TID、进程启动代次、调用序号、monotonic 时间，metadata 仅含 `shape`、`stride()`、`storage_offset()`、`dtype`、`device.type`、`is_contiguous()`，以及已有 `num_tokens` 整数；可按配置限制 rank 和观测窗口。不得读取 tensor 内容、prompt、权重或把指针写日志。
- 原样调用原方法并保持异常、返回值、参数语义；不调用 `contiguous()`、同步、额外 NPU 算子、额外查询设备状态，不修改目标源码。必要元数据访问是否触发同步/分配、图捕获是否执行 wrapper，须用目标版本源码和实际验证，不能事先保证零扰动。
- 定长、非阻塞、有限额记录；使用明确的 per-process 计数和标志记录溢出、采样、异常、截断、进程崩溃以及窗口启闭。缺失、没有 final stats、fork、进程复用时必须标 UNKNOWN；不可把没看到当未发生。

在这些条件下，入口元数据可以验证某次传入 view 是否连续和 stride 是什么，**不能看到该函数内部 `gathered = gathered.contiguous()` 返回的真实 storage 身份**，也不能证明跨流读何时结束。wrapper 不是纯旁观：它改变 Python 调用栈并产生计时和记录开销；编译图/graph replay 或替换方法可能绕开它。若不能保证调用原样转发或无法验证覆盖，应停止并报告 UNKNOWN。

生命周期需分级：可通过已记录的 producer/consumer 代码、现成 stream/event 路径与进程内观测构造候选；Python 对象 weakref 释放不等于底层 storage / allocator 可回收或 NPU 已完成读取。不能增加同步事件或人为保留强引用以制造“安全”证据。真实 storage 别名、跨流完成、graph replay、重用和写后读顺序，目前**尚无非侵入式已证接口**；是否可只观察程序原有 event/record_stream 调用，需要先核实这些调用能否被安全截获、事件生命周期和跨线程关联。未经证明不要把结果标为“生命周期安全”。

## 第二部分：worker 服务时间和排队是两个量

- CANN 9.2 `aclnn_copy.h` 确认两段式 `aclnnInplaceCopyGetWorkspaceSize(...)` 和 `aclnnInplaceCopy(...)` 声明；`libopapi.so` 动态符号表中存在这两个名称。仅在独立诊断容器、实际函数可观测且 ABI/转发经过 CPU 模拟和独立复核时，才可考虑在真实已有调用周围记录 PID/TID、单调时钟、进入/退出、状态及有界候选 token；不额外调用 API、不解引用 tensor/workspace、不阻塞，不把 executor 指针当永久身份。两段调用和 Slice_Tiling 的对应仍要证明。记录到的是**host API 的服务区间**，不是 enqueue 等待、NPU 执行或可节约时长。
- `NPUQueue.h` 声明 `Repository::Enqueue/Dequeue`；`AsyncTaskQueueInterface.h` 声明 `QueueParas.correlation_id`。当前只检查的 `libtorch_npu.so`、`libnpu_profiler.so` 动态符号中未发现 `Repository::Enqueue/Dequeue`；不能从头文件就假定它们可由 LD_PRELOAD 拦截，也不能擅自按内存布局读取参数。可能有其他库/内联/隐藏/动态解析路径，当前均未查明。两种 ACLNN API 的符号存在也不证明这次运行路径经过可拦截符号。
- **真正队列等待**需要同一 enqueue/dequeue 的已验证 correlation token、各自真实线程、同一已校验单调时钟、进程/队列生命周期与 ID 复用规则；若程序已有且可验证的队列观测接口无法以可接受开销关联，结果保持 UNKNOWN。绝不以 Python wrapper 时间接近 ACLNN 时间、stream 指针相似或逐次顺序为 ID。graph replay、跨线程转交、丢事件都要单独统计；出现缺失就不能宣称全量。

## 将来执行的客观前置与对照

1. 核验本人 baseline 容器 ID/image/健康，16 芯片实时 PID 归属和本人服务无在途或排队；发现他人占卡则暂停。以原有跨尝试共享的容器内互斥、watchdog、双份已验证恢复身份、经验显存释放条件和 health=200 恢复护栏运行独立 runtime；不覆盖先前日志。环境配置、量化模型、权重、引擎及代码 hash 均冻结并记录。若护栏不能通过，不停放原服务。
2. 先做 CPU-only wrapper/缓冲/溢出、异常透传与重复 ID mock；原样返回、无张量值记录；确定真实拦截是否存在后再讨论服务实验。不能因为 CANN 符号存在就认为可测。开关默认关闭，专用输出路径、字节上限和丢失标志；仪器异常必须不改变原始函数返回，诊断失败仍恢复本人 baseline。
3. 对照至少包含 A：无 profiler、无额外 wrapper 的同负载，B：只启用目标 metadata wrapper。若 queue 的可靠入口经另行验证，再加 C：在 B 上启用最小 worker 观测。维持 32K/1K、并发 8、实际 90% 前缀命中、同请求、same image/权重/采样参数与预热；记录成功数、query/hit、prompt/output token、draft、排队和异常。间隔交错 A/B（必要时加入 C），使用多轮及每轮结果而非挑单个最优；硬件繁忙、频率/温度、其他任务、prefix 状态、脚本窗口以及输出一致性须报告。
4. 将 A 的端到端 TTFT/TPOT/吞吐作为无 profiler 参考，将 B−A / C−B 的分布和噪声作为**仪器扰动估计**，不是优化收益；同 rank、同前缀缓存/请求阶段、同 shape 和捕获覆盖条件才讨论候选。没有严格 request/correlation ID 不把两组不同时刻事件拼成同一次。资料不足就仅给各自观测范围，不报推断性倍率或收益。

证据/复现：`python3 run_low_perturbation_inventory.py` 只在本人 stage 内低 CPU 优先级读取两份日志及头文件、查询少数库动态符号，不打印日志行或 tensor 内容。输出 `low-perturbation-inventory.json` 记录文件 hash、被检查符号范围与来源 SHA；完整既有采集结果和局限见本目录 `EP8-ENGRAM-COPY-QUEUE-PROVENANCE.md`、`EP8-ENGRAM-LOW-STACK-COMPARISON.md` 和 `agents/fusion/ENGRAM-LOCAL-RECOMBINATION.md`。头文件/符号只说明静态可见性，不能替代运行时调用证据。
