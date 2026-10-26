# Astra High：Run290 覆盖异常独立归因审查

2026-09-26；只读检查Run290、Run289产物/服务日志、bench.py、更新后的Loop065 runner。未操作NPU/服务/源码。

## 结论：并发客户端污染已证实，不能归因H003或cold

Run290的312/300 tok/s与2个Runtime cohort不是干净冻结c12实验。独立发现同APIServer pid313558日志有120条POST /v1/chat/completions，实际来自Run289、Run290两套各60请求：

| 客户端 | warmup48 monotonic时间 | bench12 monotonic时间 |
|---|---|---|
| Run289 | 4064193.490312 → 4064351.584836 | 4064352.103592 → 4064393.152725 |
| Run290 | 4064194.062373 → 4064351.585364 | 4064352.145610 → 4064393.153754 |

两套请求区间合并最大在途24；Run289“已结束启动失败”的目录后来在08:21生成完整warmup/bench产物。bench.py的Semaphore(12)包住完整stream生命周期，一个实例不能解释24在途。日志48个source ports、warmup阶段Running16+Waiting8与两个c12 clients一致。

所以：
- 这是120混合请求流中仅2个fixed c12 handoff，不能说干净60请求五波里无故丢三波。
- Run289旧容器shell/health等待流程在Run290新服务健康后继续提交，是高置信生命周期污染链；具体已退出进程无法再直接取栈，但产物时间和120POST形成独立证据。
- 无需用“冷启动”“helper慢”“随机admission”解释这轮覆盖异常。它们对其他干净Run是否有影响仍unknown。
- immediate all8 [12,4096] BF16→[96,4096] pad0与16个FULL Runtime报告仍是局部capture/replay smoke证据；120混合负载下无法估算其产品吞吐收益。
- Run283/287有不同诊断同步，不能用其5/5单独证明H003损害handoff。先排除已证客户端污染。

## 已加active marker：有用，但不是完整生命周期隔离

当前diag/profile/control runner创建OUT/active，外cleanup先删除；容器循环每次health前检查，bench前检查。这个修复能阻止仍处于health等待的旧shell看到后继服务健康后再发请求。

剩余具体竞态：
1. marker检查与启动bench之间仍有TOCTOU；bench已经启动后不再检查marker，删除marker不会停止其子进程。
2. 重用同RUN_ID会重建相同active路径，可能让旧等待者复活；marker应包含每次launch唯一token，不只检查存在。
3. 外层被SIGKILL/连接异常且trap未执行，active可能残留；仅存在不证明owner活着。
4. 不同OUT各有active，不能阻止两套合法runner同时启动；6次NPU idle检查不是原子资源锁。
5. health只验证8080可用，不验证它是该run拥有的实例；应检查记录的服务PID/启动身份或实例token，并在自身服务死亡时立即退出。
6. cleanup调用全局stop，若旧runner仍活到后继服务期间，仍可能停止后继服务；停止动作也应以实例ownership为前提。

建议最小增强是“唯一token+明确子PID/进程组生命周期”，不必发明新调度框架：cleanup取消本run容器等待/bench子进程并等待退出；下一runner启动前确认其无存活后代。单实验全局锁/原子owner记录可避免两个runner竞态。只清NPU worker不等于清掉等待health的client shell。

## 最低成本隔离验收与下一实验

先不用NPU做runner控制验证：在诊断等待循环中撤销active/token，确认容器子shell退出125且不会在随后出现的health成功条件下启动bench。另验证撤销时已启动的假bench子进程可被cleanup确实终止；这只测试已发现控制路径，不重做模型实验。

真实下一Run保持同immediate补丁与产品配置，仅修复客户端生命周期：
- 启动前保存所有相关host/docker shell与bench进程树，确认无旧run客户端；等待loop、bench PID均属于唯一run token。
- 服务端POST应严格60，客户端monotonic区间合并最大在途12；其他run目录不能新增bench产物。
- 每phase保存真实request id/服务实例连接证据；Runtime覆盖按真实all8报告计数，理想40个rank-cohort，不能用长度success替代。
- 未满足60/12隔离则该Run INVALID，不继续归因kernel。
- 若隔离后恢复5/5，客户端污染解释获干净重复支持，再进行H003 delayed/原A0局部profile与数值gate。
- 若隔离后仍不足5/5，再启用已有handoff reason/shape计数，核对num_reqs、scheduled_tokens/per-request token、prefill、admission、request membership；届时才比较A0/immediate而非先改max_num_seqs或推测cold。

Run290不支持H003 REJECT，也不支持KEEP；它要求修复实验隔离。当前正式性能和双层Bound不因这轮混合负载数字改变。

## 更新：RUN_TS实例检查已加入（审查后续）

复读diag/profile/control新版：health成功后调用_owns_healthy_service，检查某个pgrep -x vllm进程的/proc/PID/environ包含本RUN_TS。这直接阻止Run289旧shell接受Run290服务，修复了上文第5项的主要已知问题。

剩余边界：
- 检查只在首次health成功时执行，bench12前只有active检查；两phase前重新核identity成本很低。
- 只证明“存在匹配RUN_TS的vllm进程”，未证明它就是8080监听者。单实例/全局锁假设需成立；否则绑定监听PID更严谨。
- 当前 tr NUL→newline | grep -Fxq 在pipefail下可能因grep提前退出造成tr SIGPIPE而假失败；建议单进程GNU grep -Fzxq对/proc/PID/environ匹配，或完整读取后匹配。
- marker check→bench启动竞态、已在运行bench不会被marker取消、SIGKILL残留、重复RUN_ID和全局stop误停后继实例等问题仍如上文；RUN_TS核对不会代替子进程清理。

这些建议用于收紧实验隔离，不改变Run290双客户端污染归因。
