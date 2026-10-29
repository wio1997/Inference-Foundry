# DSpark 状态投票回读候选

候选从已接受版本独立复制，保留全 rank 的 MAX 投票和 CPU 读取前的同步。开启开关时复用已有锁页内存作为回读目标，并复用设备事件等待；开关默认关闭，保留版源码未改。

在 172 上使用真实 HCCL 的 16 rank 小型对照，每 rank 预热 16 对、测量 256 对，交替排列原版与候选；覆盖三项状态的组合与不同发起 rank。所有 rank 均得到预期结果，缓冲区及事件保持复用。

| 每 rank 耗时统计，再取 16 rank 中位数 | 原版 | 候选 |
| --- | ---: | ---: |
| 测量值中位数 | 410.406 μs | 381.028 μs |
| 测量值平均数 | 441.093 μs | 482.122 μs |

各 rank 中位数比值在 0.905–0.937；平均值方向相反。这里的 CPU 计时包含通信及等待其他 rank，不是纯回读耗时；也未包含实际模型，不能据此计算端到端收益。测试覆盖有限状态组合，不能证明全模型正确性。候选暂不启用，不视为已验证性能优化；后续只有发现明确原因和收益空间时再考虑完整模型测试。

原始文件：`/data/tiankuan/wio/910c-deepinfer-172/diagnostics/zos-vote-d2h-collective-20261030/rank*.json`；聚合证据：`/data/tiankuan/wio/910c-deepinfer-172/evidence/zos-vote-d2h-collective-audit-20261030.json`，SHA256 `3e3fe58dbe2852b23851ff295ad851e07b1fecda0fbe01c465e2c34b99538fb3`。候选文件 SHA256 `6dcd2297fc01d02a3fe396bfe56e1bb106cf3fef1fe38f7ab9e57dbe536e255e`。
