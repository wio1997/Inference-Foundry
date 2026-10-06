# checkpoint164 — Decode critical path and two concrete code paths

本阶段没有新增代码级性能 KEEP。已有两项可审阅最小patch；H5比H4新增了可重复的相同23-token完整自然EOS请求收益，正式标准workload/SLA产品验收仍未完成。Current=None；两项均未加入active stack。研究已从checkpoint162的unknown推进到跨rank因果、具体源码和correctness→matched A/B→完整请求证据；没有新增NPU profile、大规模Run、参数扫描或kernel修改。

Run249 profOFF的265.290943ms/token与profON的336.029329ms/token不同。下面的设备/host数字来自ON，不能缩放拼成OFF265ms的互斥预算。8输出token只有五轮target+MTP，各轮78target attention/75target MoE和1draft attention/MoE。MTP两token成块，TPOT不是逐tokenITL。

## 逐轮实际依赖与最晚供给

新的[离线逐轮归约](round_readiness.json)用同机host167时钟、rank0设备embedding边界与1901个group/op/ordinal完全匹配的HCOM操作。慢rank发生切换，不能固定以D0或D15单rank代表整个请求。

| target+MTP轮 | HCOM数 | 最晚设备入场rank | 与最晚host launch一致 | 主要晚到rank的设备embedding→D2H extent | 该rank非event task union |
|---|---:|---|---:|---:|---:|
|1|317|D13:249，D7:46，其他22|315/317|D13 486.102ms|112.459ms|
|2|396|D13:285，D15:85，其他26|395/396|D13 568.150ms；尾部切到D15|179.008ms|
|3|396|D15:394，D11:2|394/396|D15 603.685ms|47.175ms|
|4|396|D15:396|396/396|D15 577.074ms|47.052ms|
|5|396|D15:396|396/396|D15 577.171ms|46.946ms|

每轮必要链：Scheduler scheduled work/KV metadata→Executor广播→ModelRunner准备→target layers（TP projection/reduce、SFA/DCP attention、EP routed/shared MoE）→target sample+MTP draft→default-stream依赖/D2H→worker response/全贡献者KV聚合→Core commit/SSE/下一轮。第一轮远程KV命中后走compact KV gather，后四轮packed Q gather/remap/local SFA/DCP output+LSE merge；这不是D完整prefill fallback。

五轮必要通信为TP800reduce、380MoE gather、10target/draft LM-head gather；DCP395gather（第一轮79KV，后四轮316Q）、316all-to-all merge；EP380dispatch+380combine在MC2内，另计于HCOM。MTP必要通信/算力已含其draft路径。算子计算不改。详细生产者/消费者见[主归因](CRITICAL_PATH.md)。

## 分离后的可证事实

| 项 | 已有证据能说明的时间/依赖 | 不能冒充的量 |
|---|---|---|
|模型计算|第三轮D15 named-math union36.355ms，另有未命名任务；target/draft模型依赖完整|理论必要计算floor、OFF耗时|
|通信本体/peer等待|1901次最晚入场→final end总32.572ms；start skew总2008.906ms；1896次最晚host==最晚device，最晚launch→device中位24.825us|重叠总和不是wall预算；MC2不含在1901内；不能删peer必要等待|
|event/barrier|第三轮早D0 event union609.603ms，慢D15仅3.470ms；输出copy-stream覆盖整条主流链；1901中无显式HCOM barrier|600ms冗余barrier；host barrier absence；没有TASK的显式wait不能填设备时长|
|host producer/queue/runtime|D15 exposed late-launch902.406ms = preEnqueue747.341 +Enqueue→Dequeue94.418 +Dequeue→CANN60.647；D13为510.710=416.999+62.754+30.956ms|747ms全部可删Python；Enqueue start不是精确queue push|
|当前model scope|D15 preEnqueue中MoE531.027ms、MLA173.377ms、外部42.937ms，按最短active scope归约避免嵌套加总|下一enqueue名作为前一gap原因；inclusive API加总|
|Scheduler/Core/Executor/ModelRunner间隙|D15 D2H→下一device embedding4.392/7.105/6.706/6.700ms；within-layer晚供给反复出现|这几ms全部是Scheduler；各模块entry/exit未导出，精确四分法仍unknown|
|profOFF CPU|最小Run252 all16 main95.6–98.2% scheduled CPU；实际SHM最近读1s spin与runtime主动等待会计入|纯Python frontend预算；sched_schedstats=0不能推runnable wait|

表内跨流union重叠，不相加。最大已定位区域是**模型内部eager host供给**，主要落在MoE。largest globally removable OFF budget仍unknown；已有source/raw足以做两项具体代码实验，不足以声称全剩余时间都能优化掉。[扩展归因](ATTRIBUTION_EXTENSION.md)保留currently-active scope、event stream及CANN workspace证据。

## 已完成最小patch链

H4 list-output gather的16次slice copy+final split是最大已证明冗余的具体路径：D15 ON removed-scope127.203ms、局部gap交集69.691ms。必要HcclAllGather保留，private output+equal positive512B shard才走直接写入。240CPU/320native/32real-model字节检查通过。短TPOT降低6.42/10.32%，完整3-token首对近噪声，正式INCONCLUSIVE/PARKED；Run250失败原样保留。

H5 shared-expert overlap=false时四record/三wait同流依赖：ON scope133.520ms、局部交集32.119ms；这不能按133>127推H5更优。五文件patch保留true跨流依赖与必要内部同步；native176和64worker witness、原始SSE/ID/chunk/KV一致。完整23-token两对D generation TPOT229.346→218.544、226.595→219.210ms/token（4.71/3.26%）；D wall下降263.214/183.561ms，超过观测baseline drift68.956ms。P对总PD下降贡献71.079/2.239ms，单独扣开。见[完整H5裁决](H5_DECISION.md)。限域代码收益已证，正式SLO/动态workload未验收，不因API无关backlog否认它，也不把它升为产品Current。

## 剩余最高价值源码问题与现场

剩余最大具体观察为MC2 dispatch/combine frontend：D15最短active scope交集123.193/111.406ms，inclusive203.617/187.673ms；producer原CANN workspace query合计仅19.921ms，不能解释整个frontend。Pinned op-plugin源码含七个新输出的分配、descriptor/optional参数和EXEC_NPU_CMD；当前不能证明其中哪个可安全复用，动态batch/专家分发、异步combine和输出生命周期必须保留。下一研究继续现有源码/raw按allocation→descriptor→executor依赖审查，不做workspace参数扫描，不预选C++缓存patch、graph翻转或全控制链重写。SHM spin不能凭高CPU占用成为新的性能主线。只有一个真正决定性的缺口需要新增诊断时才另冻结最小动作。

15:48:54Z只读fresh核验：P249 root1916718/start303049910；D253 root1117497/start304582938；两health200/idle、各16owned NPU worker；67相关original源码SHA全部一致；253controller/phase已退出。D selector0和五原磁盘源码恢复，内存保留stock语义诊断wrapper；H4不叠加。不要按旧D249/250/251PID操作或重放spec。产品完整API/动态workload/SLA仍开放，这不是项目完成声明。
