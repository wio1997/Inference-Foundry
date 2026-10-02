# GLM — HANDOFF

2026-10-02T06:00:58.433519+00:00 checkpoint；持续自主执行。Mac仅SSH，研究/代码/raw全在166/167。branch glm5-3-autonomous-20261001，活动点[GLM-OPT-0002](records/points/GLM-OPT-0002/point.md)；历史按[source_notes](records/points/GLM-OPT-0002/source_notes.md)和Run索引恢复。

- Current：None。Run43有限C2四条81932→61440/245760有效输出/106.761171TPS；TTFTP50=4.6069s FAIL、TPOTP50=17.84494ms PASS，非KEEP或稳定容量。原stock C3 SLA失败/C2尚未验证，不把不同TP32结果当stock闭环。
- 当前Run55 running/deploy；唯一controller PID1526068/boot6d9cf06f-a02d-49ef-b7fe-cc456a7b2025/start266619261/specc83785adcbbc4e4c21c3abe9239e3a749e1b54002af6b489a1d1abebbd44d847/20pins。真实Zcode launchVALID；实时state/monitor/raw权威，checkpoint不是实时状态保证。
- Run52 REJECT：D0在主模型KV初始化之后，MTP drafter SFA metadata申请154MiB workspace OOM，尚未Graph。D1未观测自身OOM。P32ready、D32权重加载/plannedP81933 D150484；0请求输出。完整栈修正GPT早先preKV判断。Run51默认capture4的geometry错误确实发生在KV之前；52capture6已通过CPUgeometry且推进到KV后。Run50 rawKV OOM也不是硬件界。
- Run54 REJECT：128builder实际零workspace，64worker完整receipt；D已进入K5FULL6Graph capture warmup的dummy执行，asyncHCCL/RUNTIME申请内存失败。当前op名HcclAlltoAll/stackbatch_matmul_transpose不隔离真实allocation/domain；D1没自己的OOM。0请求输出；20pins/native9/owner/env审计有效，非硬件界/Graphfit/PD证明。Run55降低原生通信buf继续验证，实际收益与吞吐unknown。
- Run54任务workspace guard只消除目标SFA未使用的继承MLAbuffer：五原生sourcehash/AST校验，构造限定scope，意外runtime读取立即失败；普通MLA/非目标模型原分配。CPU实际P/D/nativeproposer构造正反例通过；160432128B/builder为CPU名义字节，实际物理收益unknown。算子/vendor/KV布局不改。原生drafter config保留targetGLM并设runner_type=draft，不能用draft_model_config整替测试。
- CPUv1导入顺序错误、v2错误draftfixture均INVALID冻结；v3VALID。Run53GPTfixedsitepath误替换在CLI预检拦截，未清理旧52/启动模型/推理/发信号，INVALID留证。54按字段改engineID与插件版本，固定现场glm52-pd不变；清理目标实际52，不是53。
- 55保留P.9GiB/K1eager/81933与D1.6GiB/K5FULLcapture[6]/max6/144384，batch1024/maxseq1，每role DP2TP16DCP16EP32，32物理NPU共目标64worker。P/D角色HCCLNPU63000/63100与HOST62400/62500起各64端口；KV62000/62100各32，RPC62200/62300/master62250/62350，每scheduler唯一engineID。四actualCLI/API/source预检、原54 exactboot/start/argv/idle确认后唯一controller联合清理，仅任务资源。 新HCCL_BUFFSIZE768→256MB同时两角色；普通PG原生200MB/DPformula不改，mc2不设PGoptions使用env。修改已分配通信buffer需要两角色重建。
- 若ready，55仅11个新nativepilot：四P→D组合、两D短local、D全输入localfallback，目标7public352输出+4Phelpers；第二来源可能已有prefix命中，不能声称全冷allshard传输。真实dualfit/Graph/PD语义/gateway/state/稳定容量待裁决。53/54均不恢复旧queue。
- 下一证据依赖：先裁决55原生驻留/Graph及pilot终态。若成功，以新唯一controller保留权重，使用独立cache_salt与return_token_ids比较同输入PD路径和D本地完整prefill控制；Usage/finish/DONE计数本身不足独立KV语义证书。不要先排旧队列。 pd_semantic_probe.py已写但GPU未执行；salt/hash/nativeRequest/PrometheusCPU正反例v2有效，首版catch错类INVALID保留。
- NativeKV生命周期源码CPU：Dtasktracker done先于PACK、Pfinished先于ACK，schedulerfinish_requests跳过已finished请求；Dfirstoutput或abort已finishedP不能证明所有Pshard已释放。无实测泄漏/时序损失。kv_both原生仅配置时recv-only且无完成归约，RFork复制到新tensor，不作共享HBM证书。
- Run46可复用native工具+STORE1状态合同320有效输出，sameAPIepoch proxy重启、previous/retrieve/cancel/background typedSSE/私有600journal；非状态复制/故障恢复/完整框架或稳定容量证明。nativeReduceTrue44必填city丢失REJECT，后续False修复验证；无量化质量比较。
- MooncakeSDK import134 teardown已有核心符号证据，但heap根因unknown；task native_acl_lifecycle仅ACLinit→原生CLI→finallyfinalize，各实际config exit0；无vendor/operator改动或exit跳过。AtomicMQ+metadataV2组合sourceguard保留。
- GPT研究代码裁决，真实/usr/local/bin/zcode --prompt负责执行监控归约，Job→Result→bridge；launchVALID≠fit/KEEP。必要清理/重启已授权，只任务资源，不再审批/等待继续。

166入口：ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166。167socket失效时从166 ssh -o BatchMode=yes root@172.16.10.167。repo /data/tiankuan/wio/Inference-Foundry；现场 /data/tiankuan/wio/glm52-pd/deploy；容器glm52-single。现场三份操作文档已读，其历史不要覆盖新的事实/用户授权。nativeHTTP禁proxy，Zcode保留私有cwd /data/tiankuan/wio/glm52-pd/deploy/private/zcode-relay-work/provider；凭据不打印/发布。
