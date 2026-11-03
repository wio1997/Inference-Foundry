# Engram metadata 独立绑定审计 — 2026-10-29

对象为 169 的诊断运行 `001d14496b2146bba25dae0f27c780b5`，仅离线读取其任务目录下已有的 progress、observer status 与 metadata 日志；没有发推理请求、操作 NPU 或修改目标实现。审计脚本 `audit_engram_binding.py` 在 stage 容器执行，结果为 `engram-binding-independent-audit.json`；原日志 SHA-256 是 `b32f03b3aca61939fbf76e56a6c7faaa0d1d8a9c4d225515f03e72082c44cfbf`。

- 客户端 progress 的 run ID 匹配、状态为 COMPLETE，并注明结果不是性能基准分数。日志 3003 条 begin/end 均属于 progress 的 window；每条记录本身没有 run ID，因此运行归属是通过 process-instance、PID 与 status 交叉绑定，而不是通过日志字段直接验证。
- 19 份 status 具有 19 个互不相同的 process instance，均与 run/window 匹配，闭窗字段均为该 window。日志中有记录的 16 个进程，均与其中唯一一份 status 的进程身份和 PID 匹配；另有 3 份 status 对应进程没有留存的 begin/end。进程个数不等于 rank 覆盖证明。
- 1513 条无重复 key 且含有效 metadata 的 begin 均是 NPU int32 非连续张量，stride 为 `[48,1]`，offset 为 0 或 24；其中 826 条与唯一 returned end 匹配。另有 687 条仅存 begin，664 条仅存 end；配对缺失不能解释为函数失败，也不能证明硬件执行结束。
- 日志末尾有 1 个 1 MiB 上限标记；17/19 status 标记 incomplete。19 份 status 中计数器求和有 `log_lock_drop=3184`、`status_drop=2`，但 observer 存在非阻塞丢弃、上限和可能不完整的状态快照，这些数值不能换算成全量调用数、抽样率或实际请求分布。
- 因此，结果只支持这次窗口中留存样本的布局观察，不证明所有 rank、图重放或整个负载都相同；也不证明直接使用该 view 的数值正确性、异步生命周期或性能收益。该采集不适合作为吞吐、耗时或算子极限数据。
