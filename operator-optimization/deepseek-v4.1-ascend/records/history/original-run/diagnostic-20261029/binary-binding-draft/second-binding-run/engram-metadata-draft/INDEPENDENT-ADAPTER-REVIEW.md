# Engram adapter 独立静态审查

审查对象为本目录五个 `engram_adapter_*.py` 和上级目录 `remote_guard.py`；均保持硬禁用，没有启用脚本、启动服务、操作 NPU 或修改实现。下述问题是解除禁用后的具体条件风险，不表示已经发生。

## P0：Python 优化模式会移除关键保护，可能误停同名其他容器

`engram_adapter_start.py:38–46,117–126` 用 `assert` 检查 baseline 的完整 ID、镜像、NPU 占用以及 lease，然后将名称查出的 `baseline['Id']` 传给 `docker stop`。远端调用 `python3 -c` 没有显式清理 `PYTHONOPTIMIZE` 或拒绝 `sys.flags.optimize`。如果远端进程环境存在 `PYTHONOPTIMIZE=1`，Python 会删除这些 assert：例如原 baseline 已被删除且另一容器占用了相同名字，`docker inspect BASE` 返回该新容器，固定 ID 检查被删除，后面会停掉新容器；恢复脚本固定的是旧 ID，也不能自动恢复这个新容器。`engram_adapter_recover.py:37,47–50`、`engram_adapter_supervise.py:58–61,126–130` 等同样有安全前置条件写成 assert。CPU 测试自己的优化模式防护并不约束实际 adapter 及其远端进程。

要求在解除禁用前将所有影响容器身份、外部占用、锁及路径/lease 的 assert 改为不可被优化删除的显式检查，并在本地、远端及容器内解释器都明确拒绝优化模式；用受控 fake Docker 在 `-O` 场景证明错误 ID 和外部 PID 仍拒绝操作。不能通过仅给当前启动命令 unset 环境变量代替移除安全 assert。

## P1：idle 检查与停服务非原子，可能中断新进入的请求

`engram_adapter_start.py:121–126` 在读取健康及 running/waiting 指标为 0 后，仍要读恢复记录与 lease，再执行停止。若其他客户端在检查完成后、stop 前提交新请求，该请求可在没有再次检查或流量接入屏障的情况下被停止；owner fence 只协调该套诊断脚本，并不阻止服务外部客户端提交。不能仅凭 polling 或“检查时为 0”声称不干扰其他使用者。需要能证明拒绝新请求、排空已有请求并冻结路由的授权接入屏障；否则不停车。

## P1：watchdog 将停滞进程当健康，baseline 可长期无法恢复

`engram_adapter_watchdog.py:19–22,30–36` 只检查 `os.kill(owner,0)`，即便 owner 被 SIGSTOP、陷入死锁或进程存在但不再推进，也会持续刷新 READY/lease，始终不进入恢复。例如 baseline 停放后暂停 owner 进程，watchdog 一直看到 PID 存在，因此不会接管。PID 复用也是同一接口的另一限制；本机 Mac 休眠、宕机或掉线时，controller 与本机 watchdog 可能一起失效。需要进度心跳、具界限的失效接管和能在本机不可用时工作的独立恢复方案；接管必须继续使用同一个 owner fence 并核对身份，不能以计时超时强行干扰未知任务。

## P2：中止 docker exec 客户端不证明设备工作已停止

`engram_adapter_supervise.py:64–81` 遇到外部 PID 或超时后只 terminate/kill 本机 `docker exec` 子进程；该操作本身没有等待容器内 `engram_client.py` 和已提交给服务的请求完成的协议，也没有证明 NPU 工作已结束。此后 `RESTORE` 会尝试停止本次诊断容器，但可能出现停止超时、锁丢失、SSH 断开等未知结果。原 mock 仅验证 `Popen` 对象收到 terminate，不验证容器内进程或设备请求的生命周期。恢复前应以精确容器 ID 确认诊断容器停止、实际进程及设备释放；仍无法确认时必须保留 UNKNOWN 并暂停，不能声称瞬时排他。

## 其他边界

`engram_adapter_recover.py:44–50` 在保存身份缺失时允许以随机 run ID、精确名称和标签定位诊断容器，这是一条为 docker create 成功但记录未落盘设计的恢复路径；标签不是不可伪造的所有权证明，应在可信受控 Docker 环境中使用并对身份冲突 fail closed。`remote_guard.py` 的持锁进程检测不能阻止不参与协议的其他用户改变卡占用；启动前采样和加载、采集中轮询均非原子硬件隔离。恢复中的外部占用或无法归属 HBM 返回 deferred 是保守行为，不能把它报告成 baseline 已恢复。

审阅时主文件 SHA256：orchestrate `aadc0b08d57c4e84e7775eaabb44052fdaecd74d42bd46a167d46f1d6db17572`；start `95d41303635120865f1cc14447b10220ed13dc05c20180e249da0ef40b61b82d`；supervise `e7ab221bf5a8665d63b3808a5bb16d61e88b71bf37d5e5fda1a9172fa39f12fa`；recover `eebd398b83d652c5ad5c0fa5776f485e2952f73806cbdcee86de2856bce68b32`；watchdog `d49b685b946e986af995f07721ee1b424c51a8167a2ad2f903b3d85421268b89`。后续源变更需要按新哈希重新审阅。

## 已加固版本的 P0 独立复核

此节对应 start `577159d1ef9222c5293a7db4c2b2ae9633793e9dfd8429cfa85aabfded2d6a0e`、supervise `fe5cb33c507bcdd466dc757aff4b50ea7b68a65ecb5c2a93926500625ecefa76`、recover `7683ddf248ccdf71472b973a152ee71f04fac14146fcf3ecba68ea3172804cd5`、orchestrate `bf477b3b26dd728822b421a1c725c7edff4bada7d72d4cc607fb22104565515b`、watchdog `9027100a2f2a2f0278f8e310a712180231a956cdde91504c9cefae9bc1d52d37`。五个入口依旧硬禁用。

用 Python AST 只读解析，五份文件外层没有 `assert`；7 个作为常量的远端执行片段 `REMOTE` / `HEALTH` / `STATUS` / `CAPTURE` / `SUMMARY` / `RESTORE` 也没有，并在其开头检查 `sys.flags.optimize`；5 个静态可解析的内层 `health` / `prepare` / `check` / `code` 同样没有并含检查。start 的动态 `verify` 采用合成值在内存中拼接和编译，结果无 `assert`；`cmd` 会为实际容器内 Python `-c` 脚本加前置优化检查。stage fence holder、watchdog lease 有单独注入的检查。`start` 的完整 baseline ID、外部 PID、创建后身份、恢复记录和 lease 已改为显式 `if` / `raise`，不会随 `-O` 被删除。

**构建流程的残留风险：**`generate_adapter_drafts.py` 仍包含用于生成的原始断言片段，它会重写五份 adapter；随后必须另运行 `harden_adapter_drafts.py` 才完成显式检查转换。因此重新生成会覆盖当前加固版本、重新出现原始断言；目前仍有硬禁用，不能因此触发服务操作，但解除禁用或交付前必须核对新 SHA，并在每次生成后强制完成转换与复审。已确认当前的嵌入常量和动态 verify，不代表对未来任意生成片段的自动全覆盖。

本次只读源码及内存编译，未调用服务、NPU、Docker 或重复测试。`test_engram_client_cpu.py:104–117` 包含片段在 `-O` 与 `PYTHONOPTIMIZE` 下拒绝的测试代码，但该代码本身不是本次执行结果；它尚不证明真实服务器的环境、Docker 生命周期及竞争条件。前文 idle 检查与停止的竞态、watchdog 无进度心跳、外部用户占用非原子检查和本地执行客户端退出不能证明设备停止等风险不因 P0 修改而消失。

## 心跳及恢复版本复核（入口仍禁用）

本节审查 SHA：controller `027ec8bf8a265053d3d8433161d630b02cd7c2ca511058cfb3168412bc81c004`；watchdog `9ac720a373e6273fa4be7162cf8cb18e72b280e339b278ac3e3ec6828204b661`；recover `06a4938fee106a5c16aa839f10168f8abdf8c7181700ca33c75ea6a87115cbeb`；liveness `9affa5b4ca0260877df274f52f661f2222afd4bb49bb7c47ed470d9c6523ca04`。源码已由 PID 存活判断改为心跳、绝对期限及身份核对；下列缺口在这些 SHA 下仍存在。

1. **P1：失联时使用旧快照可能漏掉当前子进程。** `engram_adapter_watchdog.py:49–65` 失联后使用上一次 `FRESH` 的 `last` 中的 child，而不是刚读到的 stale 心跳。具体序列：上一次轮询发生在父进程新阶段刚开始、child 仍为空；随后 controller 持久化新 child 身份、放行 child，且在 watchdog 下一次读取之前被停止；120 秒后 `last` 仍没有新 child，watchdog 跳过它并终止 owner，新 child 重定父后无法按原父子关系验证和安全终止，可能继续执行或持有远端 fence。最小修复：对最新 stale 快照分别验证 run、owner 出生标识及 child 身份/直接父子关系，不因时间过期而丢弃可验证的 child 身份；信号后还要确认子树退出，身份或存活不明时记录 UNKNOWN，不能假称接管成功。仍需处理 owner 已退出后无法可靠重验亲子关系的场景。
2. **P1：组 leader 退出不代表组内所有进程结束。** `engram_liveness.py:62–71` 发组 SIGTERM 后，只有组 leader 的 PID/出生时间仍可验证才发 SIGKILL；若 leader 已退出而成员忽略 TERM 存活，`signal_owned_group` 的失败被忽略且返回 True。幸存者可继续保留 SSH 或工作，妨碍 fence 释放与 baseline 恢复。最小修复：跟踪并确认受控组/子树成员的实际退出；无法安全认证剩余进程时返回 UNKNOWN，不能盲目对未知进程组发信号，也不能以该函数 True 声称完整清理。
3. **P1：本机进程组终止不能证明远端操作已停止。** `engram_adapter_watchdog.py:60–76` 终止的是 Mac 的本地进程/组；代码没有协议证明 SSH 断连后 169 的远端 Python 和已提交的 Docker 命令立即结束。远端可能仍持 fence 或完成此前发出的操作。恢复端正确地重新竞争同一把 fence，但必须将锁超时、远端命令结果和真实容器状态视为 UNKNOWN，持续按身份与占用验证；不能仅凭本机进程退出宣称安全接管。Mac 休眠/离线时 watchdog 本身也可能不可用，缺少远端独立恢复保障。

`test_engram_liveness_cpu.py:24–25` 名称为 40 分钟 load，但调用默认 `supervising` 阶段；`starting` 的 1000 秒期限并未被该例覆盖。现有测试用 mock 检查身份和信号，不证明上述进程交错、SSH 脱连或真实容器状态。`engram_adapter_recover.py:25` 中的远端恢复在已获共享 fence 后核对诊断容器 ID、run 标签及停后状态，发现设备其他 PID 或无法归属显存时延迟恢复；这些核对是保守保护，不证明不参与本 fence 的其他用户无法在检查间隙进入。请求排空竞态也仍存在。此次仅静态审阅，没有操作服务、NPU 或修改实现。

## 最新 watchdog 修复复核：仍有恢复链 P1

复核 `engram_liveness.py` SHA `c0c296ca7e821cfa43cf558f10bfab6b5880cbffbcb0bb86c50d8e2e4c4d3edc`、`engram_adapter_watchdog.py` SHA `12f1b445f83fe0d71408cacd3e3d98087c352020cdeb8f9ae9e33907647a206f`，入口仍硬禁用。上次 stale child 漏识别在 owner 与最新 child 均可验证的条件下已处理：watchdog 冻结 owner 后重新读并认证最新快照。leader 已退出、组仍存在时现在返回 UNKNOWN，不再误报退出；这避免了假成功，但尚不能保证清理完成。

1. **owner 已死亡时 watchdog 不恢复。** `engram_adapter_watchdog.py:52–59` 将任何非 FRESH 状态送入 `validate_takeover_snapshot`；后者在 `engram_liveness.py:36–43` 要求活着的 owner 的身份可查。baseline 停放后如果 controller 被 SIGKILL、正常崩溃或进程已经结束，验证返回 False，watchdog 直接退出，不到 `watchdog.py:78–88` 的恢复循环。最小修复：以可验证的进程身份区分已死亡与身份未知/复用；已死亡时不对不明子进程发送信号，保留未确认子树信息、通过同一远端 fence 和明确身份检查持续尝试安全恢复；身份未知时持续 UNKNOWN、告警和独立监督，不得直接失去恢复能力。
2. **冻结后身份或清理失败会把 owner 永久留在 SIGSTOP，且 watchdog 退出。** `watchdog.py:60–72` 在发 SIGSTOP 后，一旦冻结后快照失效、child 的 stop 结果 UNKNOWN 或触发异常便 `SystemExit`，没有解冻，也不进入恢复。最小修复：设计有明确所有权的冻结状态机和 finally；若 owner 身份仍经验证且恢复运行不引入并发危险，安全解冻并交还；身份无法证明时不盲发信号，仍应由独立 supervisor 保留监控、告警和受 fence 保护的恢复尝试。必须覆盖 owner/child 退出恰好发生在快照重读之间的失败分支。
3. **SIGCONT 和 SIGTERM 之间 owner 可以继续运行。** `engram_liveness.py:85–86` 先恢复整个组，再重新核验后发送 TERM。若 SIGCONT 成功而 TERM 核验或发送失败，则返回 UNKNOWN，但是 owner 已恢复，`watchdog.py:68–70` 随即退出；即便 TERM 成功，恢复到终止之间也可能继续推进。接管过程须把 UNKNOWN 当作 owner 仍可能执行，不可声称它已冻结或退出；应采用不会短暂放行任务动作的中止/隔离协议并用失败注入验证。
4. **controller 丢弃 UNKNOWN 的 child 清理结果。** `engram_adapter_orchestrate.py:49–61` 忽略 `stop_owned_group` 返回值，finally 无条件将 child 身份清空，随后 `79–80` 进入恢复。子进程可能仍在运行或远端命令尚未退出，而后续心跳丢失其身份。最小修复：只有确认组退出才清空；否则保留身份并报告 UNKNOWN，协调同一 fence、诊断容器及设备状态复核，不能宣称清理完成。

测试新增了 stale 快照与 leader 消失 mock；这两项不覆盖 owner 已死亡、冻结后身份变化、CONT 成功而 TERM 失败、或 UNKNOWN 后身份被清空的完整状态机。此轮仅静态审阅；未操作服务、Docker、NPU，也未修改实现。

## 最新 P1 修复复核：仍需处理阶段切换及冻结证明

此节审查 `engram_liveness.py` SHA `c1e5e5ba0182e6c46e1ee7c20003df347b0e6e76b27fb4b6821d1d85a3dcc0ae`、watchdog `796ddbc9f5e974a07a0f3112d84cd3b0809961453f86e68f2209803161811c90`、orchestrate `e56288f8e7a6e24ec811fc3da2e63bc249d21a945766e506a3933c17383dbce0`。入口仍 disabled。

上轮四项逐一核对：① owner 及其组都已消失、当前心跳可信、child 身份/退出可验证时，watchdog 可进入受 fence 保护的恢复；并非所有 owner 死亡场景都已解决。② 冻结后失败进入 finally，尝试只对身份匹配的 owner 发 SIGCONT，并继续重试；身份无法验证或发送失败时仍可能保持冻结和 UNKNOWN。③ 终止已认证组改为直接 SIGKILL，消除了先 SIGCONT 再 SIGTERM 的特定放行窗口；lookup 到信号发送之间的身份竞争仍存在。④ controller 在 finally 中只有确认组不存在才清除 child，结束阶段的 UNKNOWN 会被保留，但如下阶段切换仍可能覆盖。

**P1：starting 成功后阶段切换仍可丢失存活子进程身份。** `engram_adapter_orchestrate.py:56–63,26–27,79–80`：若 starting 进程 leader 返回 0，但同组其他进程仍存在，finally 保留 child 身份，`run_phase` 却仍返回 0；调用下一阶段的 `phase('supervising')` 无条件覆盖 child 为 None 并启动新进程。最小修复是每个阶段离开前都必须确认该组不存在，否则保留身份、停止阶段推进并交 watchdog；不能只在最终 `finally:82–83` 检查。

**P1：发送 SIGSTOP 不能单独证明停止已生效。** `engram_adapter_watchdog.py:69–75` 在系统调用返回后立刻重读快照，没有检查 owner 实际处于 stopped 状态。如果信号尚未生效时 controller 又放行新 child，watchdog 可能用旧快照中止 owner 而漏掉新 child。最小修复是在重读前通过进程状态接口确认同一出生身份的 owner 确实 stopped；无法确认保持 UNKNOWN，并覆盖该交错顺序的失败注入。

**可用性缺口：不可读 heartbeat 阻止自动恢复。** `watchdog.py:58–60` 在判断 owner 是否已消失前先要求当前快照绑定，heartbeat 损坏、被删或读取失败即持续 SNAPSHOT_UNKNOWN，即使本人 baseline 已停且 owner 已消失也不进入恢复。不能据此猜测不存在子进程；需完整持久化受控子进程登记，并由独立仲裁器持续检查 fence、容器和设备状态及告警。Mac 离线或休眠造成本机 watchdog 不运行也仍没有远端独立保障。

本次只读源码，没有执行服务、Docker 或 NPU，也不将 CPU mock 当作真实故障恢复证明。

## 阶段切换和停止确认的定向复核

复核 orchestrate `a772d7a6d49063ec9dca24aac2baa309cfbe6e38847d512f7e4e39d97a55f720`、watchdog `8f33a072a40fff979710a4829242854e83203c7d91d3ce98e1a7347dcf97021f`、liveness `c43f0e7ce434523e87839d97a48371cca78361affc8c429765d17478c8af4a78`。三份入口仍 disabled。

阶段身份覆盖问题已修正：orchestrate 的 `phase:26–29` 拒绝带旧 child 身份进入下一阶段，`run_phase:61–70` 仅在组确认不存在时清除身份，否则抛错不返回成功。此前 starting 成功但组仍在导致被下一阶段覆盖的路径不再成立。

停止确认问题也有针对性修正：watchdog `69–76` 发 SIGSTOP 后，先调用 liveness `80–90`，在每次状态观察中核对出生身份、UID、session 和组，等待 Darwin SSTOP 状态后才重读冻结快照；未确认则保持 UNKNOWN 并在 finally 尝试只解冻已认证 owner。源码审阅可确认调用次序和失败处理；真实线程、SSH、远端容器状态不能由此推定，既有测试记录包含本机无害子进程实验，此次未重跑。

heartbeat 损坏、缺失或无法绑定最新身份时，watchdog 继续保持 UNKNOWN，当前无法保证自动恢复；不应依据旧快照猜测子进程。详情及前面指出的限制仍适用。本次只做定向静态复核，没有操作服务或 NPU。
