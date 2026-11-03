# 隔离 kernel 加载链记录器及启动草稿

这里的记录器仅供独立诊断使用，未向真实服务注入，未运行新的 NPU 任务。默认执行启动、监控或恢复脚本只打印说明；必须显式传入 `--execute` 才能进行其操作。所有具体动作应在核对自己的容器、卡占用与脚本后执行。

`recorder.c` 按 CANN 9.2 实际 `acl/acl_rt.h` 原型在自有 stage 容器内以 gcc 11.4.0、`-Wall -Wextra -Werror` 编译。它记录传入二进制的预调用 SHA、API 成败、句柄、线程、stream、名称/entry 和程序原本发生的 stream ID、last task ID、capture info、graph execute 调用；不会额外调用这些设备接口，也不会记录二进制原文、路径、张量或 prompt。预调用 SHA 并不证明 runtime 最终读取的字节与之相同。具体关联限制见 `GRAPH-CORRELATION.md`。

CPU mock 覆盖四线程、错误返回与 errno、无效指针、真实嵌套调用、graph/stream API 及 fork。`cpu-mock-result.json` 是最近运行的结果；原始 mock trace 在 169 相同任务目录 `mock-run.6D9TsB/trace.jsonl`。共享日志的默认字节上限为 16 MiB，仅约束该记录器的协作写入；四独立进程的 4 KiB 测试实际写 3553 字节，末尾有唯一 `trace_cap`，证据在 `cpu-cap-result.json`。每进程也有 100000 条记录上限。超大二进制、不可读地址、无效名称或不可用 AF_ALG 标记 unknown。

为避免等待记录锁，线程和进程争用时会丢事件。四线程测试中 669 条入日志，后续记录的 `dropped_since_last` 可以报告部分丢失，末尾没有下一条时仍可能有未报告的丢失；不能把无标志当成捕获完整。fcntl 锁是 advisory，外部非协作写者、短写、磁盘/网络文件系统错误可破坏日志，故运行时必须使用该任务专用文件并核查完整性。记录器带来 host 时间扰动；`RTLD_NEXT` 无法解析原函数时无法转发，dlsym 与内部绑定、graph replay 的覆盖也没有得到实证。

`build_launch_drafts.py` 从已有诊断 workflow 生成 `start_binding_recapture.py` 和 `supervise_binding_recapture.py`，不执行 NPU 工作。启动草稿使用原基线容器的确切 ID、image、空闲及 NPU 占用检查，复制独立 runtime，编译单独的 recorder 并只为新容器设置 `LD_PRELOAD`，不改原框架或全局环境。监控草稿采集后尝试停止该诊断、恢复 baseline。`recover_binding.py` 提供独立的恢复与 health 检查：核对诊断 ID 和标签，只处理本人容器，其他人的 NPU 进程存在时不强行恢复抢占；仅 Docker start 成功不算服务健康。

这些脚本是可审查草稿，除了无操作模式检查外未实际运行：两段启动/监控之间以及远端断连、进程被杀或 daemon 故障时，自动恢复不保证执行，需按恢复脚本及状态记录处理。已有空闲指标和 NPU PID 查询是快照，并非阻止新请求或其他任务到来的资源锁。详细条件和目标层绑定仍需的证据见 `ISOLATED-LAUNCH-DESIGN.md`；原设计中将全局日志限额列为待解决的段落应按本文件的已实现情况阅读，其他覆盖和恢复边界仍成立。

## 后续补充：停机前监督

审查后加入 `orchestrate_binding.py` 与独立 `watchdog_binding.py`。总入口先启动 watchdog，得到 READY 后才发起诊断；watchdog 通过自有 stage 容器刷新带唯一 run ID 的租约，启动代码在停止 baseline 前要求同一 ID 的租约不超过 20 秒。主控退出后 watchdog 独立调用恢复助手，并在恢复受阻时安全重试；正常路径也检查 baseline health 后才结束 watchdog。启动记录中的 `inference_semantics_unchanged` 已改为 `null`，因为尚未完成等价验证。

`test_orchestration_mock.py` 通过 CPU-only 替身检查了正常与启动失败的就绪/恢复顺序，静态检查租约早于 stop，所有脚本和内嵌远端程序可编译；没有执行任何服务或设备调用。watchdog 自身被终止、Mac 掉电、Docker daemon 异常、租约检查后其他任务抢占等仍无法保证自动恢复；有他人占卡时恢复助手会等待而不抢占。手动恢复需从对应 launch 状态读取 `binding_run_id` 并设置 `WIO_BINDING_RUN_ID`；只针对该 ID 和明确容器身份恢复。以上仍是待实机验证草稿。

## 安全审查后的修复

启动、采集和两条恢复路径现在通过同一个远端 `owner-fence.lock` 序列化；远端进程持锁直到退出，避免启动 SSH 仍执行时 watchdog 同时恢复。启动、停止均使用已核验的容器 ID；诊断容器还携带当前 run ID 标签。launch 状态采用临时文件加原子替换，先保存单独的 run ID 身份备份。恢复可读取备份，或在缺少状态与诊断容器时直接检查本人 baseline；baseline 已运行则继续 health 检查，未运行时必须严格确认无占卡后才按固定 ID 启动。

统一 `remote_guard.py` 要求进程表完整覆盖 8 板；缺表、未知行、PID/格式错误、缺板或末尾截断都失败关闭，不当成空卡。CPU-only `test_fencing_parser_mock.py` 已验证两进程互斥、空/有进程/残缺表、无身份基线已运行、备份回退及按 ID 停止、原子状态和备份。另对当前 169 的实际 `npu-smi info` 作只读解析，完整覆盖 8 板并发现 17 个 PID；这些数字只描述检查时刻，不表示当前使用权。未进行服务和 NPU 采集。

远端锁不等于硬件资源预留；锁只协调本套脚本，其他任务仍可随时申请芯片。SSH/CLI 超时或 daemon 故障也不能保证正在进行的 Docker 操作已完成，必须检查容器状态；不要将 mock 通过当作实机恢复验证。

## 更新：非零 HBM 与容器内锁

本次改为 stage 容器内的子进程持有共同锁，进程 stdin 关闭时释放；两条实际 stage 容器 CPU-only 持锁进程已验证互斥及 EOF 释放。远端脚本在操作前检查 holder 存活，不在 SSH 宿主机创建锁文件。进程表为空并不能证明显存已释放；判据、真实负面样本、来源及 48 MB 数据驱动余量见 `HBM-RELEASE-EVIDENCE.md`。该余量每次根据历史和本人当前残差重算，是经验运行条件，不是物理安全保证，也不假设 HBM 必须为 0。

## 恢复记录与成功条件

停基线前，两份状态文件分别经临时文件原子替换、文件和目录同步，再读回校验 run ID、本人容器 ID 和逐卡 HBM 参考；校验失败不停车。两份文件都不可读时：基线如果仍在运行可以检查 health；如果已停，缺少经验参考，恢复必须保守延后并由人工核查，不能当作安全可自动恢复。既有两份正常记录即足以继续流程，不要求额外备份。

恢复流程的 SSH 总预算为 400 秒，watchdog 的恢复子进程预算为 3000 秒，以容纳锁等待、容器操作、释放观察和健康等待；这些预算并不证明 Docker CLI 超时后 daemon 的操作已经结束。恢复延后会产生非零结果；只有独立 health 检查返回 200 才能标记基线恢复成功并结束 watchdog，失败时继续受控重试。CPU mock 已覆盖两份文件读回与损坏拒绝、无参考时延后、恢复失败非零和 watchdog 保持重试；没有启动 NPU。
