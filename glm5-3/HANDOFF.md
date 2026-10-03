# glm5-3 HANDOFF

2026-10-03T12:03:33.954898+00:00 checkpoint. **GLM-OPT-0002 ACTIVE；Current=None**。功能已有真实E2E证据；达标稳定容量、全可行域上界仍unknown。继续自主研究，不重放旧Run/测试队列，不按旧PID操作。

## 当前现场与唯一controller

Run177 running，唯一controller HOST166 pid1026647/start277529849，spec6f5c9090282104a2998f0d49f90e88730e14abbb4314e1519f82c6ba9ad74d41/source110。actualphase=experiment，即176同候选freshsalt复测；模型、公服、两STORE/nativeepoch不变。D0同8192t1024serial1；D1baseline4096serial9→active1024serial10→restore4096serial11，由controllerownedidleCAS。不要并起controller/重放队列。176 VALID TTFTP504000.8ms strictFAIL0.8ms，其余5SLOPASS，native32same/bothSTORE/SDK0/fullwire/peerhints已恢复；177测量与裁决等待独立audit，CurrentNone。

| 角色 | 现役身份与配置 |
| --- | --- |
| D0 / 166:9081 | root309418/start277384683；local_pp175；TP4/PP4/DCP4，PP22,20,20,16；allocated8192/private8192t1024c1serial1；epoch c0d325815110fe4974b8c4c86d650380e6fb9d4479040d5b623dd009c3529db1 |
| D1 / 167:9900 | root913510/start276824752；local_pp168；TP4/PP4/DCP4，PP22,20,20,16；allocated8192/private8192t4096c1serial9；epoch 8b9bf327313e8f223636865aaff0dd131bc049b8a0b6be160c06cc43938f1971 |
| Public / 166:8000 | HOST551964/start277430916；Run175 public/native_engines_service_entry/V11/state125/SDKinit0active；identity_observer175绑定同public/native32 |
| 私有策略 | D0新1024serial1；D1已176恢复4096serial9。后续修改须actualpolicy/SELECTED/fallbackFalse/ownedidleCAS；不要按旧D0 PID操作 |

两域DP1/PCP1/nnodes1/local16各自完整GLM，无EP/AllGather/noKVconnector。K3/KV3GiB/Graph4,8,16,32/max32/seq8/HCCL768/W8A8/nativeMLA/nativeAsync/MTP3/noSP/noDSACP/overlapFalse/AtomicMQWorker/mathguards保留。唯一修复为PP更新中每请求new_token_ids为空时不索引[-1]的CPU控制metadata guard；没有填充tensor、发明token提交、改算子或MTP校验。nativebaseSHA failclosed/16安装marker；CPU合成分支与真实E2E通过，实际empty触发和次生copy错误因果仍未证明。

physical authority：records/points/GLM-OPT-0002/runs/GLM-RUN-0175/restored下standalone_root_identities、standalone_native_members、native_engines_resident、service_config、PP_empty_guard_workers JSON；175前的D0旧身份已退役；D1原168guard16保留，均须鲜活probe。最新Public proof jobs/AUDIT-RUN175-20261003T1200Z/public_service_proof.json。
- Mac SSH166：ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166。
- 167通过166 HOST：ssh -o BatchMode=yes root@172.16.10.167；container166无SSH。
- repo /data/tiankuan/wio/Inference-Foundry；site /data/tiankuan/wio/glm52-pd/deploy；container glm52-single。
- owner /data/tiankuan/wio/glm52-pd/controller-owner.json，TTL30s/heartbeat5/watchdog2，controller与formal-test全局锁。
- actual Zcode /usr/local/bin/zcode 0.16.5；site/private/zcode-relay-work调用glm5-3/scripts/zcode_bridge.py run ABSjob.json，实际zcode --prompt→Result→bridge；模拟不替代。GPT关键研究/代码/裁决，Zcode启动/监控/归约。研究/执行/raw在服务器，Mac仅SSH。

## 最近E2E证据

| Run | 结论和audit SHA256 |
| --- | --- |
| 166 | INVALID；PP4 fit，short64完成，coldpair600s timeout无DONE。首PP0 emptytokenIndex早于PP1copyshape5×6144→16×6144，2partial不计完成。audit8731B/da85a78bd666ab75776bf49f7920d6e2de0aaefb425dff5bfe3ff8af98de2e7c。failedD1已167精确owned退役 |
| 167 | INVALID重复GET header；两个GET200/nativewire/SDK0/0新推理。owned166D1清理/NPU0/9900absent成功，新模型未启动。audit1167B/3ff24349627f4eeb165b51a6877b235cdaa0896aee24a3cb667211e3db05a9cd |
| 168 | VALID finitefunctional251/9：PP4新epoch/guard16/coldpair81932→64 TTFT23.414/21.694s/public答案-STORE-previous-retrieve59/native32/SDK0/nohelper-preempt-D0inference。auditV2 26251B/657b8cb40919dd1046372e4e84b7b2f76ed28ecf07e92399858a547bf764ee69 |
| 169 | VALID cachedc2/data2/output64/SLOREJECT；PP4prefill3.061672s，TTFTP504494ms/TPOTP5041.6ms；129/query237604/cache146432。auditV2 21226B/64ac995c011a06629552fcaba7747315bf504f152e28bcd47b72c2ec0d855a8e |
| 170 | VALID all4/output4096/c4/t4096；16385含warm1/query401468/hit292864/5complete；TTFTP508944.2/P759109.4FAIL，TPOTP5017.7/P9020.4PASS；fullCLI132.054628finiteTPS。audit20261B/35e00c609cca466fc7ecbce54c973f038df3bb6ded9b8057e39316c11e124f92 |
| 171 | 同170精确输入输出，c2/t4096，同counter；TTFTP504466.4FAILonly，TPOTP509.7/P9015；fullCLI110.594317finiteTPS/prefill3.052418s/TTFT4.378666s。audit20582B/95ea7c3094b0c182a0e8580a7c8a64ed10c9b46067f4755547bae43fc083b41c |
| 172 | 同171/c2/t1024/同counter；SELECTED2→恢复4096serial3；TTFTP503145.9/P753814.2/P904138.9/P994333.7，TPOTP509.2/P909.6ms，6/6PASS；prefill2.816603s/TTFT3.231467s/fullCLI142.769597finiteTPS。audit21233B/a6a1c27d665aa5d12dd378c1694d15538326f9d0a694bbc0df804e5b8c8b2201 |
| 173 | 172同候选freshsalt复测/同counter，SELECTED4→恢复4096serial5；TTFTP503884.2/P754164.8/P904286/P994358.7，TPOTP509.3/P9013.2ms，6/6PASS但P50余量115.8ms；prefill3.311312s/TTFT3.830334s/fullCLI118.100734finiteTPS。audit23359B/e06c78cabd201764487f034bc2260960168781329eff40a3f9b84789dbbf90ec |

按完整native IDs/bodyhash/wire/usage/length/DONE、SDKinit-final0、计数、source pins、鲜活native32/lease0/策略恢复裁决，CLI0不够。172173两个有限4K窗口all6PASS，仍INCONCLUSIVE容量/非KEEP；N4的P99不稳健，4K不替代61440或稳定开放到达。

MTP接受率170=.690360、171=.745044、172=.927329、173=.722502，TPS不能全归因PP填充/阈值。PAIR170171只读audit22016B/283212760eba41aa316ad0c20daaf8f06635ccb49f1267c1286628d94b1f79b3：sample端点Running4-4窗口246.603644TPS，Running2-2窗口194.787616TPS，仅有限观测，不是持续GPU步占用或硬件上界。

## 状态与机制复用

旧D0 resp_glm_run139_base 已随175退役，GET/previous503 beforeleaseRPC；新D0 resp_glm_run175_D0_new原native STORE32/child16/retrieve有效；现D1 resp_glm_run168_D1_new create32/child16/retrieve有效。旧D1 IDs139other/149new/156new/164new在新epoch下GET/previous503 beforelease/RPC；不复制STORE、不跨owner回放。work_seconds hints是有限coldHTTP/短decode、epoch/sourcehash绑定的粗提示，不是cached工作/原生进度预测；peer readd须全compiler字段。

已验证透明nativeAPI/router精确lease/Chat-Completion-typedResponses/tools/reasoning/usage/采样/logprobs/STORE-previous/取消/native错误隔离/drain-readd/epoch affinity/持久faultquarantine/physicalidentity observer/open arrivals；见139/147/149-151/168，不把清单缩小完整功能范围。customID并发race、STORE复制、实时native进度/精确缓存工作仍unknown；helper3未知ACK不计成功。

昂贵实验前按机制复用：
- 152：两域各2、all4×81932→61440完整245760，全32active/fullCLI316.546509finiteTPS；TTFTP506264.3FAIL/TPOTP5011.4/P9011.9PASS（5/6），不是当前PP4 D1-only界。
- 146：开放48×21→4096/两域/196608，offered409.6TPS，晚期397–415TPS/backlog14–16；TPOTP5035.74FAIL，非稳定证明。
- 147148：变长12请求/粗work_seconds/私有预算减半，短输出延迟与TTFT取舍，无全SLOKEEP。
- 159160162165：当时PP2/DCP8/cacheblock1024下threshold8192无收益、CPUtokenization约.22s、PP38,40无prefill收益，不能永久否定PP4/DCP4。
- CACHE-MTP-CPU：hiddenstatebackoff不可移除；PP4有效DCPblock512/kernel128/LCP73738条件预测73216hits每请求；PP2有效1024预测72704，实际counter权威。
- PP4-FULLCONFIG-CPU：PP边界22/42/62合法HFfull，nativeinflight4，budget8192；CPU接受不等于NPU fit。
- 32 EP32/大KV高预算MTP临时OOM、79 DSACP/SPmath失败仅当时配置，不盲重放。
- nativeiteration_tokens_total是frontend computedprefill+gen，不是SchedulerOutput/GPU步；queue不含frontend/coreinbox，TTFT−prefill不等于纯GPU或tokenization。

## 下一问题与记录

175终态audit已VALID/functionalmigration251/9；native32健康idle/SDK客户端finalize0/无helper-preempt。沿两域各c2 PP4真实E2E研究；仅D1私有threshold可ownedidleCAS1024后恢复，D0原1024不变。WorkSeconds旧coldhint不能假定cached批次2/2；可使用已实现missingrate全域countfallback并finally完整compilerpeer恢复，明确路由条件变化；唯一controller排程。没有固定额外步骤、审批或全复现补齐门槛。

原manifest/source/raw保留；GPT终态overlay manifest_record_v2.json/checkpoint_terminal_state.json/reduction_brief.json/summary记code+Run+结论。Git权威分支glm5-3-autonomous-20261001，已同步HEAD0e2938688f45e1a572725866d52d1a9c16b07583/tree8fd563e82eae34a46cb38fea27d970c60e336c5b，checkpoint132记录171172173174终态/PAIR。下个checkpoint133记录175176代码/终态与后续活动snapshot；活动JSON固定快照；ownedbytes/tree/commit/branch/index/origin CAS，无force/无关改动。

旧历史全文见[checkpoint131 HANDOFF](https://github.com/wio1997/Inference-Foundry/blob/31dbc0a6a54b683c2922f1d58c3161deac49c98b/glm5-3/HANDOFF.md)。当前点records/points/GLM-OPT-0002/point.md和Run/job保存后续结论。Job编号时间是标识，actualstage/UTC权威，不按预先命名时间猜执行。原AUDIT168无Result、169隐藏缓存相等门槛失败均保留；新只读V2有效，不覆盖或重跑推理。

2026-10-03T12:03:33.954898+00:00: Run175 VALID functional migration: D0 exactownednative137 retired, automaticoldpublic168 D0 quarantine/D1 survivorSTORE nativewire preserved. NewD0 TP4PP4DCP4/22,20,20,16/allocated8192t1024serial1/nativeguard16 root309418/start277384683/epochc0d325815110fe4974b8c4c86d650380e6fb9d4479040d5b623dd009c3529db1; D1native168 root913510/epoch8b9bf327.../8192t4096serial7 unchanged. Pilot2short21→32+2NEWcold81932→64=192/query163906/cache0/4success/fullnativeIDs-wire-usage-length-DONE; coldTTFT24.7314/25.3265s, short6.1886/6.5269s, differenthost/newepoch/threshold andruntimehistory notisolatedgain. Correct4/42/GLM_OK_731 +newD0STORE32/previousSSE16/retrieve59 =>251/9complete; D1allnewcounter0; oldD0GET-previous andretiredD1IDs503beforeleaseRPC/no replication-replay. Public175HOST551964/start277430916/V11/state125/SDKinit0active/newHOSTobserver; oldpublic168exactretiredSDKfinal0. CPUactualfullconfig/6phasesexit0/source116/native32freshsameownedhealthyidle/allclientsSDK0/helperpreemptuncredited0. Audit26424B SHAf76ba4a4743dd37a2f34d31803cf42663d16a0bb22f66ceb6935ac25d292d097. Capacity INCONCLUSIVE/CurrentNone/no KEEP/formal61440/globalstablebound. Next twoPP4domains eachc2 under1024 threshold/cachedall4four4096 exactworkload; preserveboth currentSTOREs/nativeepochs andaccount2warmup/actual2each, avoid cached WorkSeconds coarsehint 3/1 routing mistaken asbalanced.

2026-10-03T12:13:38.867923+00:00: Run176 VALID dualPP4 cachedfour4096 c4/countfallback actual2each/SLOREJECT. Same175D0+168D1/native32/public175/epochs/bothSTORE/source107/SDK0/fullIDs-body-wire-usage-length-DONE/nohelperpreemptuncredited. Exactall4input152 plus2warmup/nativeTokenizerValidation; gen16386/query475208/cache292864/6success, each8193/237604/146432/3success; nativeRunning2each/Waiting0/KV.247089 observed. D0same8192t1024serial1; D1ownedidleCAS4096serial7→1024serial8/nativeSELECTED8nofallback→restore4096serial9. ExistingmissingD0rate fallback alleleligibleactive_count/fullnativegroupfields retained; logicalD0fullcompilerpeer+hints restored finally/no model-publicoperatoroperations. TTFTP504000.8ms strictFAIL0.8ms/P754369.2/P904374/P994376.9; TPOTP508.8/P909.0msPASS/5of6. NativeD0/D1 meanprefill3.312610/3.306467s/TTFT3.893869/3.890182s/decode35.977498/35.939195s/queue.189/.160ms; fullCLI72.430304s/226.203661937finiteTPS. MTPacceptedfractionD0.939960/D1.924424/perdraft2.819879/2.773272; noisolatedthroughputgain/stableglobalcapacity/full61440/KEEP/Current. Compared174singleD1c4TTFTP507469.7 and172173singleD1c2 P503145.9/3884.2 explicit2domain/twowarm/countfallback/cacheorderednativeMTP trajectories differences. Audit33028B SHAd86065de068124465c7cd22f0595915a27c7f11707268d53bd42ef418437c991. Next matchedrepeat sameexactall4/output4096/c4/t1024/countfallback/twowarm/newSalt, addressnearstrictSLO boundary/knowntrajectoryvariation without roundingPASS or assumingstablegain.
