# GLM — HANDOFF

2026-10-01T21:50Z checkpoint，持续自主执行。研究、代码、执行、原始日志全部在166/167，Mac只作SSH入口。真实clone /data/tiankuan/wio/Inference-Foundry，分支glm5-3-autonomous-20261001。恢复根/局部AGENTS和本文件，再按问题读[活动点](records/points/GLM-OPT-0002/point.md)、[源证据](records/points/GLM-OPT-0002/source_notes.md)、MISSION/PLAN/REUSE/RECORDING。不要恢复旧测试队列。

- Current：None。尚无满足合同的重复KEEP；达标稳定容量与上下界unknown。正式Run21四条81932→61440严格原生提交245760输出，100.124194TPS；TTFT P50 6.003388667s和未舍入TPOT P50 18.006747585ms均FAIL。单窗口/四样本不证明稳定容量/P99。
- 任务：完整GLM专用推理框架，动态请求/可变负载/PD与并行可行域；不改算子计算，不做模型/量化精度比较。GPT研究/代码/裁决，真实Zcode --prompt负责执行/监控/日志归约。唯一controller用.controller.lock/.formal-test.lock及PID/boot/start/heartbeat排程，只操作本任务资源。
- 现场：/data/tiankuan/wio/glm52-pd/deploy，glm52-single。1669081和1679900是同一个DP2/TP16/DCP16/EP32 group，两物理成员必须同时存活；不是独立副本。Stock PDproxy8000自Run24停用，诊断gateway8002仅有限窗口运行。
- 最新终态：Run34/35 completed、controller均已退出，健康原生Run34 DP2/TP16/DCP16/EP32、budget16384/gmu.80留驻；权威为Run34 adopted_model_identities.json及现场PID/boot/start/argv。Run34性能INCONCLUSIVE，Run35新coupled fault-domain gateway功能有限E2E有效；详情见末尾。下一研究候选TP32/DCP32，尚未新启动，不恢复旧队列。


## 最近终态与下一个问题

[Run29](records/points/GLM-OPT-0002/runs/GLM-RUN-0029/summary.md)：保留Run28未执行请求的精确健康组，33attempts/31completed/23161提交输出，原生400/cancel零信用。真实Zcode REDUCE-DP29-20261001T2004Z-v2 VALID，82748B SHA dd43d2e97781e8716cf9d70f50ffdfa875e85e9e54361da95897f333641f14a3。动态9088/121.8479870040901s=74.584736TPS，低于Run26的88.900785；缓存计数相同，MTP轨迹不同，无代码收益/KEEP。冷81932→2048 TTFT47.8756/48.2228s。四匹配prefix-tail命中71680/uncached10252，TTFT6.66–8.89s，nativeprefill6.18–7.12s，queue<=38.4us/preemption0：研究更大prefill预算的直接依据。EP32每worker权重25.53GiB、activation.35–.36、non-torch2.57–2.58、Graph.91–.92、KV24.85–24.87，nativecapacity2338614tokens；startup内存不是runtimepeak/容量界。

[Run30](records/points/GLM-OPT-0002/runs/GLM-RUN-0030/summary.md) driver INVALID：首条native Responses JSON实际HTTP200/completed/READY/Usage13+2=15，GPT误用不存在的obj.error属性后失败。已执行唯一请求2输出作为valid partial evidence保留，不重放；真实Zcode REDUCE-RESP30-20261001T2024Z VALID，3441B SHA cf61e3f9c3998f9d3338e968ccbee2466dcc0a5c8c7f1a947b1caac8a8b0bbe2。

[Run31](records/points/GLM-OPT-0002/runs/GLM-RUN-0031/summary.md) 完成20:27:06Z，controller450032/boot6d9/start263201604已死。仅余8attempts，3新completed/6输出+5预期400/404拒绝，第一Run30响应只离线复用/新信用0。真实Zcode REDUCE-RESP31-20261001T2032Z VALID，7977B SHA 2bdb1f56d13986a67006959344a116641ad9f335f584109ed41d1038f7915cba。原生Responses SSE每条9个named typed事件、sequence0..8、response.completed终态+Usage，不发Chat [DONE]。5代理lease全部释放，客户端wire与gateway原始wire逐字节/hash一致，两端idle。实际两端VLLM_ENABLE_RESPONSES_API_STORE未设/native默认0；store关闭，background400、未知/previousID404属于配置行为。当前只证明无状态短请求与拒绝合同，不声称背景/持久状态功能完备。若开启store，native各APIserver本地dict需要owner亲和；当前router首输出观察Responses仍保守unknown，提交以独立native协议审计。

## 控制原型与复用边界

metadata opt-in Worker仅替换实例的两个native源guard控制谓词，保留SP/oproj/embedding/draft/Graph、CPU collective和MC2全局max/masks/buffer。native_sync SHA3ae98d968ade0f78463b04143be79792d41c484fb938f25cbb6bf3ee8b93ee8e；determine SHA812c989f97f1a271731f4b7d1c41938014d567f541aff92aac588dbc1373af50。V2 helper SHA33bee48a7bac3b48cbfd4f86c4f626721dfb1e8cf6333ff013030da5870c30a4；worker SHAc329333f50dc57f63b6b580d008c3d0516490a2e66f50085a59676ebba567660。实际vendor源码未改。
CPU12源/dispatcher案例与native fullEngineArgs通过只证明配置和控制合同；Run29证明有限E2E但没有直接live padding shape/收益证据。Run28成功加载却因外部INFO过滤造成安装日志assert0/16失败，0推理，INVALID driver不是fit/语义REJECT；原code/spec冻结。CPUlogger-v3证实外部INFO过滤。

旧Run8/9/11的16k OOM发生在47.28GiB权重/约1GiBfree/PD merged draft，不能据此否定现在EP32约25.53GiB权重的新内存条件。Run24继承PD-only recompute flag使local kv_roleNone配置失败，0weights/0requests；Run25合法省略此flag后fit。Run25部分driver INVALID，但cold/tools valid复用；Run26/27在同组证明功能/动态/prefix，均无KEEP。Run23旧独立两端dynamic101.365TPS与新跨机EP32不可孤立因果比较。

工具插件glm_tool_contract默认缺tools/choice none保留原生文本，提供tools沿用native结构约束；解决Run19四条usage却只有三条严格有效的错误。Run20/21/25/29实际工具分支通过。prefill_aware为默认关闭的路由选项，Run15/16重复证明局部短请求TTFT约6.7→.7s，但总TPS无收益。完整历史以Run manifest/summary及point/source_notes定位，不恢复任何旧队列。

## 接入与操作

166：ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166。
167 Mac socket当前失效；从166 ssh -o BatchMode=yes root@172.16.10.167。
现场读SSH_AND_OPERATIONS.md、PD_START_AND_TEST_GUIDE.md及deploy/HANDOFF.md。native请求显式禁HTTP proxy（curl --noproxy '*'、ProxyHandler({})/trust_env=False）；Zcode保留既有互联网proxy。
CLI0.16.5的默认deepseek-flash SSE曾十分钟timeout；帮助列--settings但实际拒绝。当前真实/usr/local/bin/zcode在任务私有cwd /data/tiankuan/wio/glm52-pd/deploy/private/zcode-relay-work使用已有tiankuan-dsv/deepseek-v4-flash-0731 provider，Job→CLI→Result→bridge已多次VALID。全局配置未改，凭据private600不发布。所有失败Jobs原样保留，不冒充执行成功。

## Run32 runtime memory failure and Run34 workspace discriminator

Run32 terminal20:58:21Z/controller1001047 dead, startup/Graph/API fit but firstcold native mergedMTP DCP path OOM: allocation1.01GiB at about1GiBfree/56.12GiBallocated/56.47GiBreserved. HTTP200/SSE500/DONE/noUsage/nofinish, zero committed credit; simultaneous peer issued but interrupted, emptyeventartifact/outcomeunknown/zero credit. Twoissuedattempts,0completed/outputs, remaining stages notexecuted orreplayed. ActualZcode REDUCE-BUDGET32-FAILURE-20261001T2103Z VALID; reduction62565B SHA463e43fbc315fe1ded235e0a1d1c88ce5204506bf8edd8c138fa3deabd60625d. REJECT specific16384/.87/K5/EP32coldconfiguration, not hardware bound. Startupreportedweights25.72GiB,activation1.36,KV23.65-23.66; memory profiling did not cover actual mergedMTP peak. Both16-worker installationreceipts present; native idleDP1 controlshape local6/global16384/syncedNONE/actual6 is directlyreported. This verifiescontrolcondition wasexercised, not speedup or functionalcompletion.

Run33 INVALID cleanupdriver beforeanysignal/model/request. SavedAPIroots hadcorrectoriginalargv/boot/start, but native childenviron dropsPYTHONPATH; strict envplugin attribution failed bothnodes. FrozenRun33 retained, no16k/.80runtimeclaim. Run34 checks exactliveroot ancestry or originaltask worker installationreceipt/containerNSpid/boot/start fororphan membership, recordsidentities beforeanysignal and onlysameidentitycleanup.

Run34 uniquecontroller PID2024868 boot6d9cf06f-a02d-49ef-b7fe-cc456a7b2025/start263445794; specSHA632700d7d8625b486bb51ed36350f17c7aa13ffa2efa7fcf35e8591b49de7f1c/24pins. ActualZcode BUDGET-WORKSPACE-LAUNCH34-20261001T2108Z VALID. Samebudget16384, K5/FULL/DP2TP16DCP16EP32/nativeoperators/checkpoint/parser/communication; gmu.87→.80 expectedfrees~4.29GiB KV forMTPworkspace, actualnativeE2E nowvalid inRun34 as recorded below. Necessaryfailedcohort jointcleanup/rebuild onlytask, bounded31completedtarget+400/cancel, no oldqueue. CurrentNone/noKEEP/capacityclaim.

Source-only PREFILL-TOPOLOGY-SOURCE-20261001T2053Z-v3 realZcode VALID: sourceindex2536B SHAd2a95d3c569a29a52bd54825320d2e9c8aad692d69f1d4b9ea244475f0e2cd04. NativePCP consumesphysicalranks and prohibitsDP>1; currentAscend platform1516 explicitlyrejectsPCP>1 beforeengine. BothDP1/TP32/DCP32/EP32/PCP1/node0-1 acceptedfullnative EngineArgs world32/local16, nativeNPUWorker, sameK5/FULL/16384/.87/nativeautooperator defaults. CPUonly0weights/requests, notfit/HCCL/Graphproof. DP1 remoteCLI explicitheadless needed bysource; singleAPI/scheduler vscurrentDP2twoschedulers is real capacity tradeoff. Originalprobe driver setDP_rank0 implicitlyforcingexternalLB rejectedDP1; v2validnativePCPrejection interruptedsecondprobe, v3recordsrejectionthenchecksTP32. Allraw retained; no speculativePCPguardbypass or NPUtopologyqueue.

## Run34 workspace fit and Run35 coupled fault-domain gateway

Run34 completed21:26:33Z/dead controller2024868,31completed/23118outputs of33attempts. Real Zcode all-original-commit/source/identity/idle audit VALID, full reduction157837B SHA64f78d8b9db589e9d95be05b1c7e9b6f9e8654b06366d637940f14fce8117394, curated reduction_brief.json. 16384/.80 fits bounded E2E, lower KV19.37–19.38GiB vs failed32 23.65–23.66; startup weights25.72/activation1.36,32 installed worker receipts and actual eager local/global padding metadata observed. No operator changes/kernel timing/isolated code gain proof. Cold81932→2048 TTFT50.226/50.672s; dynamic68.8123915TPS vs29 74.5847365/26 88.900785. Four canonical matched tails hits71680/uncached10252,TTFT6.071–6.185s/TPOT20.124–29.465ms, above4s. Frozen prefix summary stale Run25/4096 limits label corrected only in curated note, not raw artifact.

Run32 genuine coupled EP32 native failure caused peer engine hang despite health200; generic independent endpoint cooldown misstates fault domain. New opt-in coupled_placement.py/coupled_gateway.py declares immutable physical execution groups and exact native owner epoch, latches whole group on native failure, preserves existing peer leases/no retry, same-epoch logical readd cannot bypass quarantine. Default groups[] retains independent behavior. Controller-owned native rebuild plus new gateway/epoch recovery; no persistent crash recovery yet. CPU COUPLED-FAULT-CONTRACTS-20261001T2130Z real Zcode VALID,35 contracts,1729B SHAf5a9024e1f1be24051e629d726a4922c6c7ec726c7d38e8c7cf998a1a0210c92; actual Run32 error payload with reconstructed SSE framing, zero new native faults/inference.

Run35 completed21:36:27Z/dead controller3242094 boot6d9/start263615421; spec0c3fcf40f5fb260f27692546d95773505094eca27a4d7720c86c261f29849fcb/15pins. Exact healthy Run34 group retained, new gateway8002 only finite window then exited. 11attempts/9completed/1552outputs; native400/cancel zero credit, real mixed chat/completion/stream/nonstream/n2, drain/readd/arrival/cancel, epoch/source/lease/nativeidle/raw upsteam wires checked. REDUCE-COUPLED35-20261001T2143Z-v2 real Zcode VALID,33549B SHA63fd1ba63dcd148108cc0be4f158f8e45e706cfc40f1f32fd62408eefca447f7. First auditor failed its own epoch serialization mismatch, no inference, original retained; v2 uses exact prepared epoch canonicalization. Healthy E2E and CPU failure-latch contracts have distinct scope; no new native fault/recovery/performance/KEEP/capacity proof.

Source-only DP4-TP8-TOPOLOGY-SOURCE-20261001T2118Z real Zcode VALID,sourceindex3227B SHAd8356b35146a6d6e308b6a54b15c066495f40454c386535db4a5bbd2db4a737a: four external ranks native full config TP8/DCP8/DP4/EP32 world32/local8 accepted, per-DP nnodes1 on two physical hosts with disjoint8-device slices. No NPU queued/fit/collective proof. TP32/DCP32 alternative accepted CPU earlier; one scheduler vs two/four is actual capacity tradeoff. Current None; next actual topology is evidence-led, no PCP guard bypass.
