# 无 profiler 隔离诊断：执行链草稿和审查边界

本次先新增 `isolated_execution_draft.py` 与 CPU 假对象测试 `test_isolated_execution_draft.py`，随后以第二次运行链为模板生成独立的真实 adapter 草稿 `engram_adapter_{orchestrate,start,supervise,recover,watchdog}.py`，以及无 profiler 客户端 `engram_client.py`。这些副本含 Docker/SSH 操作代码，但每个入口均**硬禁用**，目前不可用于启动服务；未修改旧脚本或目标推理实现，未启动、停止任何容器或 NPU 负载。独立 bootstrap、adapter 安全性、观测覆盖和真实恢复仍需主控审查，不能把 CPU 测试等同远端安全验证。

## 执行链和恢复约束

1. 使用现有 `binding-diagnostic/owner-fence.lock`，不得另建独立互斥；用新 UUID 建立独立 watchdog 及新路径 lease，并验证 lease 新鲜。预检必须核对固定 baseline 容器 ID、镜像 SHA、health 200、running/waiting 都是 0、16 芯片完整 HBM 和 PID 记录、每个可见 NPU PID 属于该 baseline 进程，经验释放参考有效；未知即拒绝停放。
2. 只在任务目录为该 UUID 创建全新 runtime 和诊断容器，身份包括精确完整容器 ID、名称及任务/UUID 标签。复用冻结 serve 的原始参数、配置、权重和执行资源，验明无 profiler 和旧 hook；唯一新增是独立、明确可审计的 Engram metadata bootstrap。映射复核只读模型挂载、可写目录严格限定本人目录；真实容器的环境、挂载、设备、端口等仍需适配器逐项核查。
3. 两份独立身份记录均先持久化再读回；停放前重新核对容器身份、参数、健康/无请求、NPU 归属和经验 HBM 参考、fence 与 lease。停放本人 baseline 后先连续确认释放和再次核查，再启动精确诊断 ID；发现非本人占用或无法归属时，停止推进，按已批准的恢复策略处理。
4. 诊断健康后，单独记录每 worker 的安装确认及实际 rank 映射；找不到时标 UNKNOWN，不能凑满 16/16。正常和异常路径都进入基于运行 UUID、双份持久化身份、标签以及 NPU 占用的恢复；若恢复无法安全确认，不停止未知容器、不启动可能冲突的服务，watchdog 继续按策略重试。只有本人 baseline 的固定 ID 再次健康为 200 且诊断恢复状态已确认，才使 watchdog 完成。
5. 草稿故意将“命令发送超时”视为**结果未知**：包括创建容器、停服务、启动服务等，不能据异常假设命令没有执行。`recover_and_verify_exact_baseline` 因此要能处理未知 diagnostic ID，按精确 UUID 标签及持久身份定位，匹配冲突则拒绝操作。

一个需处理的具体死锁风险：现有 `recover_binding.py` 自己获取同一个 owner fence；草稿中的恢复接口是在持有 fence 时调用的。真实 adapter 不能直接在持锁时再运行旧 helper，否则会互锁；必须选择受审的持锁内恢复实现，或者先安全释放并由恢复 helper 和 watchdog 重新竞争**同一个**锁。任何锁失效都不能继续做容器动作，watchdog 负责独立恢复。

## 八请求和 metadata gate

冻结客户端必须重现原 benchmark 客户端的完整预热与 `warmup.json` 前缀 seeding、采样、spec decode 和响应方式。无 profiler 的测量建议顺序：先完成原 warmup 阶段；对测量阶段的 8 个冻结 seed 请求并发同步启动，确认每个 29696 输入 / 1 输出；记录测量前指标；在独立 metadata 目录对带 run UUID 的 gate 做临时文件写入及原子 rename；随后八个冻结 32K/1K 请求经 8 线程 barrier 启动，等全部结束后在 `finally` 删除**属于该 UUID 的 gate**，再读取指标。任何 seed、请求、gate 或指标失败都保留失败状态并进入恢复，绝不能删除其他 run 的 gate。

测量的 gate 只表示主机侧观察区间；它不为异步设备完成做屏障，也不保证 worker 每次读取看到相同状态。图重放可能绕过 Python，不能以缺少记录称没有调用；进程安装、rank 对应、begin/end、日志锁竞争、缓冲丢失、cap、进程退出与关闭窗口 ack 均需独立核查。当前 `engram_metadata.py` 尚没有设计中要求的完整 window ack 和丢失计数保证，bootstrap 尚待实现，不能以草稿声明已满足这些条件。

`draft_case` 仅模拟**测量阶段**；独立 `engram_client.py` 补了待审 HTTP 客户端、前置 warmup、冻结文件哈希核验、seed、同一 run/window gate 和请求等实际代码，但尚未在容器运行。它将测量有效性限定为 8 次成功、合计 262144 prompt / 8192 output token、262144 查询 / 235904 命中、0 抢占；不相符保存为无效数据，不能改成符合。只持久化摘要，prompt、生成文本、请求 ID 不进入输出。TTFT/TPOT 如需上报必须单独按原客户端口径处理并验证，目前没有提供它们。

## CPU 验证与剩余工作

本机执行 `python3 -m unittest -v test_isolated_execution_draft`：11 项通过，覆盖占用或配置变化时不停车、lease 失败、创建/停放超时的恢复、设备未释放不启动诊断、恢复未证时不结束 watchdog、gate 开启异常或请求失败的清理、冻结输入和计数校验。测试只使用假对象，不能证明真实 Docker 竞争、进程回收、PID namespace、文件原子性、watchdog 存活或多 rank 覆盖。

草稿 SHA-256 `cb42a2882d184e3a281ec377d9795bf69d019c00486221415fc92849973e203d`；测试 SHA-256 `7dd14570f07d76db9584821ef3118662c599adb584ddd50356cc129f4abf18cd`。

## 第二版：真实 adapter 草稿和复核

`generate_adapter_drafts.py` 先核对第二次 run 的 start/supervise/recover/remote_guard 源文件 SHA，再生成独立副本；旧文件未变。启动、测量与恢复各通过原有 `remote_guard.py` 在 stage 容器获取**同一把** owner fence，调用结束时释放；后续独立恢复进程重新获取该锁，不在持锁期间递归调用原恢复 helper。新 controller 另有本机非阻塞 flock 防止同机两个 controller 并发，不能替代跨主机 owner fence。UUID 用于独立 runtime、容器名、远端 lease、身份记录和本机主要状态文件。

启动适配器在停放前核验原服务 hash、冻结 seed/request/reference 文件 SHA、baseline 容器 / NPU 占用与 idle、配置和带 UUID 的双份恢复记录；创建容器后检查设备、网络、IPC、显存/CPU 配额等 HostConfig 关键项以及原 Env 相等。仅加入 metadata bootstrap，拒绝旧 profiler 或非空 LD_PRELOAD/PYTHONPATH 冲突。metadata 为 task-owned 的 0700 路径，拒绝 symlink，并与 baseline 容器 exec 的有效 UID 比较；该 UID 不是实际 worker UID 证明，bootstrap 还须各自核查。诊断加载和测量期间每次抽样检查可见 NPU 进程归属；发现其他 PID 或检查失败即退出当前阶段、只处理己方任务。采集期间发现外部占用到下一次最多约 5 秒检查之间仍存在窗口，无法保证与其他独立用户的资源操作原子互斥，不能把轮询称为绝对排他。

`test_engram_adapter_guard_mock.py` 复用旧护栏测试于新副本：两进程互斥、坏 identity、无可归属 HBM 等通过；`python3 -m unittest -q test_engram_client_cpu test_engram_capture_abort_mock test_isolated_execution_draft test_engram_liveness_cpu` 共 32 项 CPU 测试通过，含其他 PID 或 fence 丢失时取消己方 client、gate 失败清理、嵌入脚本编译、硬禁用入口、心跳期限和进程身份检查。测试未调用 SSH、Docker 或 NPU，不证明生产守护或恢复可行。

binary_binding 子任务已完成 observer v4 的 CPU-only stage 验证，修复 install 失败回滚；清单 `bootstrap-approved-sha256.json` 已更新其 hash，但仍为 `PENDING_CONTROLLER_REVIEW`。现存源目录祖先被有权限者并发替换、实际 worker/rank 对应、图重放、所有窗口闭合、观测时不会遗漏及影响大小仍未证明；不可将当前状态当作诊断通过。独立脚本、恢复语义与冻结 workload 的真实环境匹配仍需主控审核；硬禁用不应被当作已获得运行许可。

## 优化模式及失联接管补充

控制脚本与嵌套远端检查中的 `assert` 已由生成器内置转换为显式 `if` / `raise`；宿主、远端、stage 脚本、fence holder 和 client 拒绝 Python 优化模式，启动前也拒绝 baseline 环境非零 `PYTHONOPTIMIZE`。用 `-O` 和 `PYTHONOPTIMIZE=1` 对 7 个固定远端片段做 CPU 拒绝检查。生成器先在内存硬化并编译全部副本，再原子写入每个文件，单独运行 hardener 被禁用；所有入口依然硬禁用。

新的心跳草稿每 5 秒刷新阶段进度，watchdog 只有在身份匹配且进度新鲜时才续远端 lease；120 秒无更新视为失联，启动阶段上限 1000 秒、监督阶段上限 6000 秒、恢复阶段上限 3200 秒，监督阶段可覆盖原 40 分钟加载及 40 分钟采集。新子进程先阻塞等待管道握手：controller 核验进程组/会话、记录其 PID 和 Mac libproc 微秒出生时间到原子 heartbeat 后才放行；heartbeat 停止时，watchdog 仅向同 UID、出生时间与会话匹配且是 controller 直接子进程的组发信号，再对匹配的 controller 组发送 CONT/TERM，必要时 KILL，并经原 owner fence 竞争恢复。若 owner 已退出导致父子关系无法核实，watchdog 不擅自杀不明进程，可能必须等待 fence 释放；PID 核验和发信号之间仍有不可完全排除的竞争。这是设计与 CPU mock，**未在真实 SIGSTOP、SSH 断链、服务加载或远端锁持有情况下验证**。

恢复脚本在 stop 前记录本人诊断容器 PID，stop 后重新 inspect 精确 ID 并要求已停止；若这些 PID 仍出现在 NPU 表，或其他 PID / 无法归属显存仍在，则禁止重启 baseline。用户请求的“发现外人占用才暂停”存在采样、空闲检查到 stop 以及独立用户创建任务之间的竞争窗口；本任务共享 fence 不能锁住外部用户。

本次最终候选 SHA-256：start `577159d1ef9222c5293a7db4c2b2ae9633793e9dfd8429cfa85aabfded2d6a0e`；recover `06a4938fee106a5c16aa839f10168f8abdf8c7181700ca33c75ea6a87115cbeb`；supervise `e957bd4c43b6d440e4c8855acb579b7bf1620df4af840d84eecf85d2a9b62885`；watchdog `9ac720a373e6273fa4be7162cf8cb18e72b280e339b278ac3e3ec6828204b661`；controller `027ec8bf8a265053d3d8433161d630b02cd7c2ca511058cfb3168412bc81c004`；liveness `9affa5b4ca0260877df274f52f661f2222afd4bb49bb7c47ed470d9c6523ca04`；client `2bd0d7382acc3cd8ec75a21fa8dc4a5cafbd595f666ac7134408f0c23163ef8c`。这些 hash 只标识待审候选，不表明运行获批或行为已在真实容器证实。

### 2026-10-29 watchdog takeover CPU-only review update

Execution remains **hard disabled**; this is a draft for independent review only. No container, service, SSH or NPU action was run for these changes. The bootstrap manifest remains `PENDING_CONTROLLER_REVIEW`.

The watchdog validates the **latest** snapshot against the previously authenticated run/owner birth, complete child PID/birth pair, direct-parent/session/group identity, time monotonicity and phase limit. On a stalled/deadline snapshot it freezes the authenticated controller process group, re-reads and re-authenticates the latest snapshot, then processes that snapshot's child. This addresses the case where a child was published after the previous poll. The pipe gate prevents an unpublished child from executing its payload; that statement does not prove no orphan process exists. Any uncertain identity or termination leaves the result UNKNOWN and aborts automatic takeover, potentially requiring manual intervention; a frozen controller may remain frozen. No recovery success is claimed in that case.

Group cleanup reports `EXIT_CONFIRMED` only after the OS reports the whole local process group absent. If the leader disappears while the group still exists, identity cannot be revalidated, so it returns UNKNOWN and does not send KILL. Local group absence does **not** prove remote Python, Docker, worker or NPU cessation. Recovery must reacquire the same remote fence, re-query container/worker/NPU status and verify baseline health; fence acquisition may wait or fail. This remains conditional on the unchanged remote guard and is not proven by this CPU mock. There are unavoidable process identity lookup-to-signal, external GPU occupancy and baseline idle-check-to-stop races; recovery is not guaranteed if identities cannot be authenticated or remote processes retain the fence. Do not enable without independent review and a concrete recovery procedure.

`starting` has an absolute 1000 s limit; the 40-minute case is in `supervising` (6000 s), not `starting`. Local 35-unit suite and existing CPU guard mock pass, including stale newer-child selection, stale identity refusal, leader-exit/group-survival no-KILL, startup deadline and 40-minute supervising. These mocks are not real loss, process scheduling, remote fence or NPU tests.

SHA-256: `engram_liveness.py` c0c296ca7e821cfa43cf558f10bfab6b5880cbffbcb0bb86c50d8e2e4c4d3edc; `watchdog_heartbeat.template` e9a4adf50098002384bd85b1e05bb543d0d23fc4fe0413f53445348c43c8ff7e; generated `engram_adapter_watchdog.py` 12f1b445f83fe0d71408cacd3e3d98087c352020cdeb8f9ae9e33907647a206f; generated `engram_adapter_orchestrate.py` 027ec8bf8a265053d3d8433161d630b02cd7c2ca511058cfb3168412bc81c004; generator 631cf1233b9884e098685891d0a4a3551e7c7d224c291934dac509ee7eee58e3.

### Follow-up: owner death and failed takeover (supersedes previous watchdog hashes and failure-handling text)

Watchdog remains alive and retries if snapshot, ownership or child cleanup is UNKNOWN. On freeze-path failure it attempts SIGCONT only after revalidating the original owner birth/session/group; if that cannot be proven it never signals an unknown process, records `FROZEN_OWNER_IDENTITY_UNKNOWN`, and continues watching. Exact verified owned groups are terminated by SIGKILL directly, including stopped groups; the termination helper no longer SIGCONTs before terminating. A controller cleanup no longer clears child identity unless the local group is observed absent; if not, it retains the identity and yields recovery responsibility to the independent watchdog.

Owner death is separate from identity UNKNOWN: local PID **and** process group must both be observed absent. Given an intact latest run/owner-bound snapshot, the watchdog authenticates any surviving child using its original birth, UID, and session/group (parent may change after reparenting); unidentifiable or remaining groups cause continued UNKNOWN supervision, no signal, and no recovery success. Once local groups are absent it attempts the existing recovery routine, which must obtain the common remote fence and independently inspect the remote container/NPU and service health. Local absence is not remote completion; remote fence/worker/occupancy checks still decide whether recovery can proceed. Races, orphan descendants outside the recorded process group, independent watchdog failure and lack of durable remote fencing beyond the invoked guards are not proven away. If identity cannot be authenticated there is no guarantee of automatic restoration. In particular this is not approval to enable the draft.

38 CPU unit checks and existing CPU guard mock pass; new cases cover PID/group absent vs permission/remaining group, detached owner snapshot binding, direct kill without resume, no kill for unidentified leader, and static failure-path/controller retention checks. This is **not** a full end-to-end process-fault simulation or evidence of recovery on the server. No SSH, container, service, or NPU operations were performed.

New SHA-256: `engram_liveness.py` c1e5e5ba0182e6c46e1ee7c20003df347b0e6e76b27fb4b6821d1d85a3dcc0ae; `watchdog_heartbeat.template` 662dc02f35b51cebe3e161ade80f177c2972e920ed56d67d9d596de047092747; generated `engram_adapter_watchdog.py` 796ddbc9f5e974a07a0f3112d84cd3b0809961453f86e68f2209803161811c90; generated `engram_adapter_orchestrate.py` e56288f8e7a6e24ec811fc3da2e63bc249d21a945766e506a3933c17383dbce0. Manifest remains pending and both entrypoints disabled.

### Follow-up: phase barrier and confirmed stop (supersedes prior hashes)

Every `phase()` refuses transition while a child PID or birth is retained. A normally exited leader with a still-live group causes the phase to raise, leaves the original identity in the heartbeat, and prevents both `supervising` and the controller's own `recovering` transition; the independent watchdog remains responsible for determining what can safely be done. A newly spawned child is recorded as pending before identity capture; its pipe gate is still closed. Pending or corrupt identity is UNKNOWN and may require intervention.

After sending SIGSTOP, the watchdog checks the same owner's birth, UID, session and group and waits until macOS `PROC_PIDTBSDINFO.pbi_status` equals `SSTOP` before reading the frozen snapshot. `SSTOP=4` is from the local macOS SDK `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/proc.h:151`; this is a process-state observation, not proof that remote tasks stopped. Failure keeps UNKNOWN and tries to resume only the authenticated owner. A local real harmless subprocess stop-state test and a mock of the normal-exit/live-group transition passed. Total 42 unit checks and the existing guard mock passed. No server/container/service/NPU actions were performed.

If the latest heartbeat is missing or corrupt, the watchdog does not use an old record to kill a possibly unrelated process and does not claim recovery. It retries with visible UNKNOWN; the present design has no independent durable child registry and **cannot guarantee automatic recovery** in this case. Parent identity/kill checks retain TOCTOU risk, as documented above; remote fence and health must still be checked. This draft remains disabled pending independent review.

SHA-256 `engram_adapter_orchestrate.py` a772d7a6d49063ec9dca24aac2baa309cfbe6e38847d512f7e4e39d97a55f720; `engram_adapter_watchdog.py` 8f33a072a40fff979710a4829242854e83203c7d91d3ce98e1a7347dcf97021f; `engram_liveness.py` c43f0e7ce434523e87839d97a48371cca78361affc8c429765d17478c8af4a78.
