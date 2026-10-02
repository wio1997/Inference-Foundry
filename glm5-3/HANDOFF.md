# glm5-3 HANDOFF

2026-10-02T14:35:51.962431+00:00 checkpoint。持续自主执行；Mac仅SSH；branch glm5-3-autonomous-20261001；活动点GLM-OPT-0002。Current=None；稳定容量和硬件有效上界unknown，任务未完成。

- 现场：166/167，容器glm52-single；部署/data/tiankuan/wio/glm52-pd/deploy；源码/data/tiankuan/wio/Inference-Foundry。166入口ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166；167从166用ssh root@172.16.10.167。Mac167旧socket不可用。
- Run83 completed，controller695024/start269711864已退出，当前无活动GPU controller；以83state/身份和现场controller锁为准。Zcode实际zcode --prompt→Result→bridge执行；GPT研究/代码/裁决；不要恢复旧队列。
- 驻留D79：D0=166:9900，D1=167:9901，DP2TP16DCP16EP32共享32NPU，DSACP关闭/显式enable_flashcomm1保留SP/noKVconnector/overlapFALSE/nativeMLAworkspace保留/K5 FULL_DECODE_ONLY capture[48]/Async/nativeW8A8。seq8/GPUbatch16384/KV12.8GiB/HCCL4096/STORE1。budget4096t2048c2/serial1/cohort0079。原P9081不活动；测试gateway8002由各Run创建清理。原166代理8000不视作当前活动资源。
- 必读源保持：根AGENTS、glm5-3/AGENTS、START_NEW_CHAT；现场HANDOFF/SSH_AND_OPERATIONS/PD_START_AND_TEST_GUIDE按需恢复。限制：不改算子计算、不自动转kernel、不作模型/量化质量比较；必要功能回归以真实答案裁决。
- 77/78：DSACP+DCP+SP/noKV虽Graph/wire/count192通过，原始文本重复乱码；78同API2NPU32/6数学与字面答案0/6。REJECT，有效服务credit0；原77技术audit limitedfunctional=True保留，GPT语义裁决supersedes。不是唯一根因/硬件上界。
- 79：DSACPoff+explicitSP/nativeDCP16恢复6/6真实答案+coherent短/冷长pilot，共214outputs；冷81932→64 TTFT50.005/47.584s。继承日志标签提及DSACP错误，actualCLI/source/nativeclasses明确off/SPtrue；新gpt_functional_verdict澄清，不改raw。
- 80：V4实际CLI/fullnativeResponses两STORE域、8Response提交256+8tools254=510outputs；native32/8192 budget、原wire/body、归属、SSE/replay、后台运行零HTTPleases、重启gateway/retrieve/cancel/idle均通过。API物理重启状态不复制，native并发duplicateID未GPU测。
- 81：与74同12冷输入突发/可变outputs10112，新D79/V4，242.7927s/41.6487有限TPS；两long均D0，TTFT111.079/107.924s，参考SLO全失败。计数策略把81k/31k当前prefill视为同数；first reasoning output标记正常。
- V5 CPU失败两次保留：第一次夹具self.path AttributeError；修正后实际网关a/b/a而非a/b/b。继承acquire_for_owner(ownerNone)绕过subclass.acquire。新v2placement/V6显式ownerNone→self.acquire；已绑定owner保持旧路径。CPUv3实际CLI+ASGI20/20通过，无GPU模型操作。
- 82：同D79进程域/预算/12负载，新cacheSalts/V6 prefillbytes，10112outputs/213.7548s47.3065有限TPS；long分D0/D1，TTFT69.794/89.045s。D1HTTP10/nativeRunning8/wait3而D0HTTP2/Running2/wait0；短TTFTP50从81的2.709升4.374s，参考SLO全失败。字节策略分开long但集中decode，不能提升Current。raw/IDs/wire/coldcount/SDK0/idle/epoch audit有效；两轮并发片段连贯。
- 83已审计：V6/prefillbytes真实完整Responses/tools/state同80工作包、新run83 IDs；8Responses提交256+8tools269=525有效outputs，nativebudget16/32/8192、双STORE域/工具/归属/原wire/后台零HTTPleases+restart/retrieve/cancel/SDK0/idle/epoch全部通过。AuditVALID36691B SHA7a1028d8215cefa655a29f8f51ee69e60dd827e30bf24d40bf1c0606ed535c6f。下一问题是同时考虑prefill和decode工作/原生压力；不把字节当剩余GPU时间或有限wall当稳定容量。Responses SSE phase及HTTP完成后的后台占用仍有观测缺口。
- 历史有效参照：69/70 nonSP驻留，threshold2048/cadence2改善本地decode公平性而长prefill变慢；72V2全API462outputs；73active_count10112/195.53s51.72有限TPS；74prefill_aware10112/169.21s59.76有限TPS，均非稳定容量/KEEP。43prefix-warmed81932→61440并发2/4条106.76TPS但TTFTP50 4.607s未达4s；不能比冷81k小output。PD61P256/MTP1+smallD在两机资源内第一81932请求600s无wire，发送阶段progressunknown；P KV入场修复不等于P计算可行；P无MTP尚未运行、跨PD真实ABI需按需验证。75warmup overlap fault/76projectionGraph161002已留因果边界。

完整事实/路径/hash见point.md/source_notes.md和各Run manifest_record/reduction_brief；大raw在服务器。昂贵实验前按机制复用，不全量重跑。不宣布全框架完成、稳态容量、KEEP或到达硬件极限。
