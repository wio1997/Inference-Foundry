# 已有调用的被动观测与图关联边界

本草稿使用相同 CANN 9.2 `acl_rt.h` 定义原型，仅在程序**本来调用**下述接口时转发并观察返回值；不自行调用这些接口、不插入同步：

- `aclrtStreamGetId(stream,&id)`：成功时记录 stream handle 与报告的 id。
- `aclrtGetThreadLastTaskId(&id)`：成功时记录调用线程的 last task id。
- `aclmdlRICaptureGetInfo(stream,&status,&modelRI)`：成功时记录该流的捕获状态和 modelRI。
- `aclmdlRIExecuteAsync(modelRI,stream)`：记录同一个线程内的调用 begin/end 与 call 序号；它是可观察的一个可能图执行入口，不保证代表所有 graph replay。
- `aclrtLaunchKernelWithHostArgs` 亦有 begin/end，记录同线程 call 序号及 funcHandle、stream。

不论成功失败，拦截时转发原参数、返回值、errno；只有程序成功写出的标量/句柄用 `process_vm_readv` 取副本。记录器内执行的嵌套调用被抑制，真实函数内部对其他被拦截 API 的调用允许正常透传并单独记录；CPU mock 验证了其顺序及返回值。观察带来 host 时间与调度扰动，不承诺耗时不变；`RTLD_NEXT` 找不到下游时无法转发，这是当前草稿限制。

## 可判别

1. 同一 PID、TID、无本进程/线程已知丢失、严格匹配 begin/end call 序号的嵌套事件，可以说明该段 host 调用里观察到了原程序调用什么 API、其返回值是什么。`RIExecuteAsync` 相同有效 modelRI + stream 说明在此 host 上下发了该 modelRI 的一次调用；成功返回不等于设备计算完成。
2. 若下游 API 在同一个 launch begin/end 内调用 `CaptureGetInfo`，且成功明确报告 capture 状态与 modelRI，那么可报告该 host 调用范围内观察到了相应流和 modelRI 的信息。此为同进程事件关系，仍需 API 状态语义及 stream 身份一致。相邻的 `GetThreadLastTaskId` 只能在调用方/接口语义已证明其对应该 launch 时解释，不能仅凭先后将它当物理 profiler task ID。
3. 现有 Python sidecar 的 `call_begin/call_end` 带 PID、native TID、`time.perf_counter_ns` 和层 label；C 记录带 PID、TID、`CLOCK_MONOTONIC`。先在相同容器验证 Python perf_counter 的实际 clock 与偏差、两个进程视角的 PID/TID 命名空间一致性以及日志不丢失。只有同线程严格包围且关联关系唯一时，才能提出 host 层范围与 API 调用的条件性包含关系；Python 异常/多线程转移、未关闭范围、时钟或日志缺失则 UNKNOWN。原 sidecar 还受 PROFILE_ACTIVE gate、调用类型、编译与 graph replay 影响，必须验证是否在实际区间生成事件。

## 不可判别

- 获取 `modelRI` 不给出 graph node 到 funcHandle 的绑定；图 replay 是整个图的触发，不代表每一次 replay 有 Python layer forward 或普通 host kernel launch。若存在多个候选节点/模型，必须用图节点元数据或其他已验证的精确映射，否则模型级 replay 不能指认某一 layer 的 task。
- `aclrtStreamGetId` 返回值不自动等同 profiler physical stream ID；`GetThreadLastTaskId` 不自动等同 device task ID；`modelRI` 也不自动等同 profiler model ID。必须有独立版本/运行实例映射，不能单纯按数值或时间就假定相同。
- `libnnopbase.so` 动态符号表引用前三项，但不等于当前 MC2 分支确实执行。`libtorch_npu.so` SHA `7e5eb430fd0f1d438e9598cd89e609332d4d1eea1c458e8f42fd01060afa47a1` 的可见动态符号表不直接引用上述图 API；`strings` 有 `aclmdlRICaptureBegin/End/GetInfo` 和 `aclmdlRIExecuteAsync` 字符串，可能以动态查找等方式使用，实际查找机制与是否被 LD_PRELOAD 接住未验证。未看到事件只能记 UNKNOWN。
- 16 MiB 共享文件上限只针对该 recorder 协作写入；文件锁非阻塞，争用丢失会在随后的成功记录中报告 `dropped_since_last`，尾部若无随后成功记录仍可能不可见。CPU mock 在 4 个线程记录占用非常高时发生大量有意丢弃；标记缺失不能证明完整。不同进程间 event 并集即使文件顺序存在，也不自动形成因果。
