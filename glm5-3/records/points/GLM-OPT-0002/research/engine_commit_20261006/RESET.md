# Performance Research Reset — 2026-10-06

身份：glm5-3-autonomous-20261001 / b23845e4411632f0369a4ead7472b5f913f3ff6a / GLM-RESEARCH-RULES-v2 / rule commit b23845e4 / GLM-OPT-0002。Mac已fetch；服务器从18955694以相同Git对象bundle fast-forward，dirty raw保留。

产品与Current：现有正确算子下功能完整GLM专用推理框架。Current=None；现役Full Replica / Complete-request Placement，两副本各自完整Prefill+Decode。功能合同引用MISSION及Run139/240/241；历史231/234/235有限完整请求均三项SLO FAIL，不能作为当前matched baseline或稳定容量。完整产品剩余Gap和可达界unknown。

Active KEEP stack：[CURRENT_PERFORMANCE_STACK](../../../../../CURRENT_PERFORMANCE_STACK.md)，为空。

只读现场：public241 V14 PID3212971/start299869850、D0 root2064352/start295229107、D1 root380291/start293474771与原身份相符；两机容器running，public8000/D0:9081/D1:9900 health200、native请求running/waiting0。166唯一controller-owner显示Run241 completed且controller不在进程表；锁文件存在但未由controller持有。public身份observer3213329仍在。167现役root树16个有效worker，NPU16卡owner对应；首次真实Zcode READONLY-RESET-20261006补齐两机native32归属和CLI 0.16.5，bridge及独立validate均VALID；后续现场Run仍需重新preflight。没有生成、重启、信号或旧Run重放。

最大代码Gap：唯一active假说H1——最老batch在合法安全提交结果已ready之后，EngineCore优先enqueue新batch是否阻挡下一有效Decode或输出。根因/最大Gap排名/可消除关键路径时间unknown；不是已证实的18ms/token空泡。旧grammar候选PARKED，旧cap/cadence扫描停止。

决定性证据：
- 安装core.py SHA6fdd067f…，step_with_batch_queue 652–687先schedule/enqueue；696–714 FIFO取旧result后native update。从当前239 plugin及真实argv确认现役，不能以root runtime同名文件判定。
- 安装multiproc_executor.py SHAceb34774…，FutureWrapper 70–100/collective_rpc354–416：result才串行dequeue之前的RPC；done=False不是device not ready。WorkerProc.enqueue_output 948–969先AsyncModelRunnerOutput.get_output，再RPC MQ入队。
- native AsyncOutput.get_output同步copy_event后构造实际tokens；V2PP next_decode_eligible_step=current_step+PP，placeholder/KV及PPHandler FIFO slot的消费者必须保留。
- Run230决定性raw terminal_queue_native.stdout bytes365794 SHAfe0993fe…，serial106行2169–2180与serial107行2245–2256：cap3几乎每CPUcall execute、cap2交替empty；明确device_completion_time=false，没有ready/commit/实际设备区间。
- Run231/234/235 manifest→audit→raw均完整16/14208，但salt、order、MTP轨迹有混杂。131.777/94.249/135.568 TPS仅方向信号。

代码假说与patch：尚无性能patch；唯一有条件候选是EngineCore最老结果有真实非阻塞ready证明时，在额外schedule前FIFO提交一次，再保留必要enqueue重叠。现有FutureWrapper.done不能作为guard；不能预先drain-first。先实现只观测的最小时序诊断，保持原始schedule、future消费者、abort/KV/MTP/多rank和输出字节。

最小matched A/B：H1诊断先关联schedule/request/execute+sample RPC顺序、worker输出copy-ready+reply-enqueue、core接收与FIFO commit、Decode eligibility及有效SSE；不以wait时间当设备idle。阳性才做一个原生控制patch，CPU correctness覆盖FIFO、异常、abort、deferred grammar、PP placeholder/KV与多rank，然后同模型/资源/负载/到达/采样/Graph/cache/观测A/B、两组或bracket噪声重复完整E2E；完整合同/SLO保持才PERF_KEEP/提升Current。

反证：有可靠证据证明ready晚于新enqueue、或commit不影响Decode eligibility或有效输出/设备关键链、收益来自batch/MTP/必要工作变化，则H1关闭；若缺ready或设备覆盖，INCONCLUSIVE，不跑大型E2E。

Astra：YES。触发Current长期None/无代码PERF_KEEP，需独立检查H1是否值得最小诊断；Review Package同目录ASTRA_PACKAGE.md。当前不选PD/Full Replica架构分叉，不做大重构。

本阶段没有新增代码级性能 KEEP。真实E2E Gain=unknown。
