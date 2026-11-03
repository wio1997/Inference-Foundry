# 最小撤销恢复草稿：锁与异常路径独立审阅

当前文件仍硬禁用；本次只读源码，没有运行服务、Docker 或 NPU。审阅对象 SHA256：`build.py` `84de0b13ea787cea16b3451fb77ce89d582fbe1fd5b5584a4460d7c007610318`；`controller.py` `b3c8c0a52102ec6b7d06b557ef385c9217d2a093ce5ae652db893f032810e9c2`；`watchdog.py` `8d5a9cdcd0f050cee158d64b3c4de8d0647799d2c93329112ce7318c199f08f5`；生成的 start `89f924de30744328a0791658d1ed3efeb93b0047b95071fa362735d536b53abd`、supervise `b90bfcdcc613aab92ae43ff55a893ece5ac1352e3315e81e6295b7808859825a`、recover `9703f71b12668f037e3581fa38d47880526fc085509089d647d602a02920ad9d`。

在共同 fence holder 始终存活、各操作遵守同一协议且文件系统正常的前提下，`build.py:9–16,30–44` 使 start、CAPTURE、RESTORE 在获取共同 fence 后检查本 run 撤销；`17–27` 使 recover 在取得同一 fence 后写撤销再恢复。锁先授予旧操作时，恢复会等待；授予恢复并完成撤销后，后来的受控写块拒绝。这是条件性顺序关系；文件缺失以外的读取异常均拒绝。CPU 测试中的本地 flock 不验证远端 Docker 异常语义。

**P1：锁 holder 中途退出时，在途 Docker 操作可越过撤销屏障。** `remote_guard.py:28–44` 的 holder 是独立的 stage 子进程；start 远端 `REMOTE` 内 `cmd()` 在 `subprocess.check_output(docker 命令)` 前、后分别检查 holder。若检查通过后已发出 `docker stop baseline`，但该命令未完成前 holder 死亡，flock 可能释放；recover 可获取锁、写取消标记、重启本人 baseline，而已发出的 stop 此后才生效。后一次 `ensure_owner_fence` 只能在 Docker 返回后发现 holder 已退出，无法撤销已到 Docker daemon 的操作。同理，进程/SSH 断开或命令超时并不证明远端操作没有发生。因此 README 已明确的 holder 失效竞态确实存在：不能宣称恢复后绝不会被旧命令改变。需建立能证明在途远端命令已终结、并对最终容器状态重新核对的机制；在无法建立有效屏障时，恢复标记 UNKNOWN，不能凭本地 flock 模拟认证安全。

**恢复有界条件而非保证。** `watchdog.py:25–44` 超时或 owner 身份未知时启动 recover，但若原持锁操作一直不释放、stage/SSH 不可达、Mac 自身离线、已有陌生用户占卡或 HBM 无法归属，恢复会等待/重试或完全无法运行。这个行为保护未知任务，但可能让 baseline 持续停放；`done`/health 不能解决本机和远端共同故障。

**证据范围。** `test_revoke_cpu.py` 只模拟三个本机临时文件及正常锁释放的先后，不包含 holder 突然死亡、Docker daemon 在响应中断后晚完成、断网重连或真实运行中身份变化。`STATUS` 和 `SUMMARY` 按当前代码是只读；未来增加可变状态时必须将其纳入锁与撤销审阅。shared fence 无法限制不参与该协议的其他用户，新请求/占卡竞争仍存在。此结论不批准启用草稿。
