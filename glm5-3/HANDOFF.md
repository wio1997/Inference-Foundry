# GLM — HANDOFF

2026-10-02T12:03:26.659576+00:00 checkpoint；持续自主执行。Mac仅SSH入口，研究/代码/raw在166/167；branch glm5-3-autonomous-20261001；活动点GLM-OPT-0002。

- Current：None。Run43 prefix-warmed/resident-observed C2四条81932→61440/245760输出/106.761TPS；TTFTP50 4.6069s FAIL、TPOTP50 17.84494ms PASS，非冷prefill或稳定容量/KEEP。
- 当前 D69两API166:9900/167:9901、32NPUworkers驻留；Run72唯一controller已completed且退出，任务gateway8002已干净停止/端口free。身份见runs/GLM-RUN-0069/adopted_model_identities.json及native_member_identities.json，不能按旧PID盲signal。
- D69配置：DP2TP16DCP16EP32/seq8/GPUbatch16384/KV13743895347B/GMU.8/HCCL4096/K5FULL[6,12,24,48]/nativeAsync/operators/STORE1。CPUV3原生阈值/调度策略，private policy budget4096/threshold2048/cadence2/serial4，globalprefillinterval2；仅限定CPU控制热更新，非任意拓扑/KV/capture热切。
- Run69/70原始wire、冷cache计数、token IDs、epoch、策略日志已独立验证。70同队列2048阈值将D1短TTFT43.77s降到3.20–3.22s，long44.39s变50.99–51.47s，跨DP短chunkgap2.44–2.46s变约1.18–1.40s；cadence额外收益小，chunkgap不冒充正式TPOT。69/70有限768/2304输出，非SLO/capacity/KEEP。
- GatewayV2修复自定义ResponseID并发分派竞态：派发前锁内持久化原生API归属，禁止跨owner覆盖；同owner nativeAPI判duplicate行为，原始body/wire/参数不改。CPU9合同通过；实际native并发duplicate-ID接受性未测。
- Run72真实完整gateway两API STORE域与8工具案例成功，462有效输出（Responses256+tool206）。previous/retrieve/cancel/typedSSE/replay/后台完成/代理重启/逻辑drain-readd/原生400-404及body-wire/lease一次释放核验通过。后台native正Running且0HTTPleases：不得据leasealone宣decode-free。代理journal只保存owner指针，不复制nativeAPI状态；物理API重启恢复未证明。
- 无现役P；旧P61/D56/D65/D67均已精确清理。Run61第一P81932 helper600s零wire/无terminalnativeerror，during-requestcontinuousprogress unknown；Run58 nativeP SFA114MiBOOM，reportedfree~500MB不证明allocator-domain/rootcause。Run56原P41/null1/free40不能容纳需要41blocks，1GiB46/free45解admission但不自动解实际HBM。均为条件失败，不否定PD或宣硬件界。
- 已安装原生PDproxy缺Responses/genericcatchall；optionalPD完整接口集成尚未实现。未来P无speculation source候选可减HBM，但尚无GPU收益；D public K5保持。算子/vendor数学/量化质量不改，zero-unusedSFA受nativehash/AST/poison限定。
- 失败版本不覆写：70audit旧policy路径INVALID后v2VALID；CPU缺CANN环境INVALID后v2VALID；CPU测试误读groups INVALID后execution_groups修正版VALID；71 controller阶段名含下划线INVALID/零modelrequests，72新Run有效。
- 下一问题：在驻留D69完整gateway上测可变context/output、动态到达、bursty负载与恢复，依据有效E2E latency/queue/KV/提交量推进controller/并行/PD研究。已有证据够则复用；不恢复旧队列，不等待继续，不因所有复现缺口未补完停研究。
- GPT关键研究/代码/裁决；真实zcode --prompt → Result → bridge执行/监控/归约，唯一controller管理共享资源。launchVALID不等fit/KEEP，Current保持None，硬件乐观界和稳定容量unknown。必要任务内启动/清理/重启已授权。

接入：166 ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166；167 socket失效时从166 ssh -o BatchMode=yes root@172.16.10.167。repo /data/tiankuan/wio/Inference-Foundry；site /data/tiankuan/wio/glm52-pd/deploy；container glm52-single。现场SSH_AND_OPERATIONS/PD_START_AND_TEST_GUIDE/HANDOFF已恢复，历史不覆盖实际state。HTTP禁proxy；Zcode私有cwd/provider/凭据不打印发布。raw/canonicalbody/GET_META设备指针binary仅服务器。详情和hash见point.md/source_notes.md、各Run reduction_brief及checkpoint_evidence。
