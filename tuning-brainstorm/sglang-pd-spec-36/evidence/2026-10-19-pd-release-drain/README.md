# M005/M031：PD 传输完成、abort 与页释放

当前做法通过 request 状态、drain ACK 和可选 deferred-release 队列管理传输期间的页。候选方向是先保证最后一个写入任务已结束，再省去 L025 allocator `.item()` 的隐式同步；本轮没有修改生产代码，也没有接受该候选。

本轮获得了可改变下一步设计的证据：**队列超时不等于 drain；通用 Future helper 与 Ascend override 的错误处理不同。** 这些是源实现与 CPU 边界实验结论，不能当作已发生的 NPU 页复用故障。

## 固定源码与采集状态

固定安装版本 `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 可从 SGLang 公共仓库读取。本轮通过 API / HTTP 获取 [13 个固定源码文件](source-index.json)，校验 Git blob SHA1 和 SHA256；R004 的 12 个文件中 8 个与已有安装快照完全一致，补齐 4 个本地缺失文件。R007 再补 custom-pool selector。

这恢复了固定版本源码闭包，**没有重新确认离线容器的当前运行状态**。旧 SSH socket 已消失，R001 在 banner 阶段超时，source payload 为空；直接非交互连通性检查也超时。没有据此推断远端服务已停止，没有重复执行 R001。

原始失败与无效执行一并保留：R001 的 loop 登记发生在 dispatch 后，见[登记修复](registration-recovery.json)；R002 因本地预检查尝试写系统 Python 缓存而失败，shell 后续仍登记/dispatch 缺失的冻结文件，见[无效 Run](R002-invalid.json)。R002 没有测试结果，计分为零。随后改成内存语法检查，并分别确认冻结文件、登记、执行；R003 使用新 Run ID。

## 释放策略：138 项检查 / 25 场景

[R003 结果](release-policy-result.json)执行 11 个完整源方法，验证既有 canonical 采集 receipt、文件内容和 body hash。tensor、collective、network、物理 free 都是显式边界夹具，没有导入 Torch 或完整服务模块。

| 条件 | 当前源码行为 | 本轮证明范围 |
| --- | --- | --- |
| device，ACK 不全，截止时间前 | 保留页、metadata 与 receiver | CPU 状态转换 |
| device，ACK 不全，达到截止时间 | 警告后进入释放 | 4 个 deadline 场景；不是写入已停止的证明 |
| host-staged，ACK 不全 | 超时后仍保留 | timeout 不替代 drain |
| host 本 rank ready，但 MIN 对端未 ready | 保留 | collective 为明确夹具，不是多进程测试 |
| 重新 arm 同一 room | 清空旧 ACK set；重复 rank 不重复计数 | 未验证网络认证或 room generation |
| 一个释放回调抛异常 | 该条目已从 retry ledger 移除，其他条目继续释放 | 失败条目的完整资源恢复仍未证明 |

见 [queue release](sources/installed/python/sglang/srt/disaggregation/decode.py#L2762) 与 [ACK tracker](sources/installed/python/sglang/srt/disaggregation/common/conn.py#L638)。`required_acks` 是 defer 时对 bootstrap 列表长度的快照；测试在之后改写列表，确认不会把门槛缩成零。物理设备页、host backup、metadata、receiver 的释放顺序仅执行到明确的资源边界。

## 通用 helper：13 项检查 / 6 场景

[R005](future-drain-result.json)运行完整 `_await_transfer_futures`，使用真实 Python Future 和一个自有 CPU 线程。测试由 Event 控制先后关系，等待超时只作 watchdog，没有性能计时结论。

| helper 输入 | deferred 关闭 | deferred 开启 |
| --- | --- | --- |
| 首个完成任务返回非零，另一个仍 running | 取消尝试后返回，running 任务尚未结束 | 等到 running 任务完成才返回错误码 |
| 首个任务 `RuntimeError`，另一个仍 running | 异常直接外逸 | 同样直接外逸；没有 drain 剩余 Future |

因此，该 helper 的“启用 deferred 就不会有写入任务越过返回点”的注释需要限定错误路径。[源方法](sources/installed/python/sglang/srt/disaggregation/mooncake/conn.py#L1030)。这里的 RuntimeError 是明确注入的 transfer-function 边界失败，没有观察到真实 SDK 或远端写入异常。

## Ascend 实际覆写：30 项检查 / 6 场景

[R006](ascend-drain-result.json)执行完整 `AscendKVManager.send_kvcache → MooncakeKVManager._transfer_data → MooncakeTransferEngine.batch_transfer_sync`，使用真实 ThreadPoolExecutor / Future。SDK、地址和 layout validation 是明确夹具，整数地址不指向真实内存。

Ascend 的直接 custom-pool 分支逐层提交 Future，遇到非零状态立即返回，**没有调用通用 drain helper**。无论 deferred flag True 还是 False，错误返回时第二层 Future 都可仍在 running。普通 SDK Exception 会被实际 wrapper 转为 `-1`，同样进入这条早退分支。因此，R005 的通用 RuntimeError 反例不能直接描述成“普通 SDK 异常穿过 wrapper”；R006 单独核验了实际转换与覆写。

源码见 [Ascend 发送](sources/installed/python/sglang/srt/disaggregation/ascend/conn.py#L384)、[transfer bridge](sources/installed/python/sglang/srt/disaggregation/mooncake/conn.py#L753)、[engine wrapper](sources/installed/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L246)。

**适用性仍有门槛。** `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 声明默认 None；selector 识别 NVLINK / BAREX / INTRA_NODE_NVLINK，无设备能力检查；另一个 initializer 可能失败后回退。manager 读取环境 selector 与物理 pool 是否初始化成功需要分别记录。没有证明正常 Ascend 配置能够或正在使用该分支。[selector / initializer](sources/installed/python/sglang/srt/disaggregation/mooncake/utils.py#L132)、[环境声明](../2026-10-19-npu-plan-war-source/sources/installed/python/sglang/srt/environ.py#L792)。

默认 custom-pool 关闭时，直接 Ascend 路径将所有 layer 合成一次 `batch_transfer_sync`，没有这些 per-layer Future。CPU 夹具核验了这个调用形状；SDK 返回后是否仍可能写入设备页，必须由 SDK 合同和原生实验确认。Hybrid MLA / PP / DCP 各有额外路线，不能从该单批场景推及全部路径。

## 调用者与页所有权边界

sender / receiver 继承关系已由固定源码补齐：Ascend sender / receiver 继承 Mooncake，manager 继承 Mooncake / Common；Ascend transfer engine 继承通用 engine wrapper。worker 按目标 session 分片到队列，每队列有一个 transfer worker；因此不能仅凭某个 abort 分支 `pop(outstanding)` 就认定同 room 存在并发写入。

worker 将 Future / transfer 调用完成作为递减 outstanding 和发送 drain ACK 的依据。异常会进入 worker 的致命错误分支；线程监督、executor 退出、远端写入完成与 outstanding 恢复仍需独立审计。receiver 的 terminal poll 与 metadata gate 也不自动等价于设备消费者和物理页均已结束使用。

## 候选方向、成本与否定条件

下一轮可单独设计：在具备实际可达性证据的 Ascend 逐层分支复用已有 drain helper，并为 helper 的异常路径明确 drain 后传播异常的语义。它属于候选代码方向，本轮没有构建、应用或接受补丁。默认关闭行为、SDK 错误转换、首个错误信息及取消语义都需保留。

新增成本是错误时等待其他写入完成；若某个 SDK 调用永久不返回，线程可能持续阻塞，不能靠补丁宣称全局资源已可回收。device deadline 仍可能先到，故单修 helper 不等于完整解决页复用合同。正常成功路径、错误暴露时间、host 容量和 end-to-end 性能都要同条件检查。

[原生合同](native-next-contract.json)保留 Ascend、PD 分离与推测解码/MTP及完整 36 机制目标。取得任务资源后，必须记录真实 pool/flags/流/SDK 与 PP/DCP/backend，覆盖成功的小 native allocator、延迟写入越过 abort/deadline、ACK、metadata/device/host free 与复用、输出和接受分布，以及同条件性能。任何释放后写入、loc/页集合/输出不一致即否定；未解释的生命周期继续 hold。

本轮分别报告 138、13、30 项组件检查，不把它们合成原生正确性或加速成绩。整体 **E0**，accepted baseline 为 null，L025 候选仍未接受，性能未测。
