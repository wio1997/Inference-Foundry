# glm5-3 HANDOFF

2026-10-03T13:47:46.213560+00:00 checkpoint. **GLM-OPT-0002 ACTIVE；Current=None**。持续自主研究；功能已有真实E2E，稳定达标容量/全可行域上界unknown。无固定扫参/补齐所有复现/审批门槛；不重放旧Run/队列。历史见[checkpoint133 HANDOFF](https://github.com/wio1997/Inference-Foundry/blob/a891090ce093669abc254f5d1b0d4e859714525e/glm5-3/HANDOFF.md)及对应Run/job。

## 当前资源与唯一controller

Run180 running/唯一controller HOST166 pid3776850/start278105307，spec0b73f1f5a0c1b120d2b59b1691016ab750be8f5eadc58c1b335c8745439f69b2/source110。actualphase=experiment；samepublic179/native32/epochs/STORE/V12/HTTPmemo，全4×81932→61440/c4/countfallbackactual2each/2warm/freshKV盐/noCPU预载；目标full245760+2warm245762，全计E2E。D0serial3不变；D1baseline4096serial15→active1024serial16→restore4096serial17，ownedidleCAS。不要另起controller/重放队列。179终态VALID，coldTTFTP504101.4失败/reuse4hit2miss3867.8全6PASS仍非稳定KEEP；CurrentNone。180真实长输出/SLO/成本/界等待终态audit。

| 角色 | 现役身份与配置 |
| --- | --- |
| D0 /166:9081 | HOSTroot309418/start277384683；local_pp175；TP4/PP4/DCP4/PCP1/DP1，PP22,20,20,16；allocated8192/private8192t1024c1serial3；epoch c0d325815110fe4974b8c4c86d650380e6fb9d4479040d5b623dd009c3529db1 |
| D1 /167:9900 | HOSTroot913510/start276824752；local_pp168；同TP4/PP4/DCP4/partition；allocated8192/private8192t4096c1serial15；epoch 8b9bf327313e8f223636865aaff0dd131bc049b8a0b6be160c06cc43938f1971 |
| Public /166:8000 | HOST3171974/start277979053；Run179/public.gateway.log/service_config；native_engines_service_entry_v12；state125/SDKinit0active；HOSTobserver179同native32 |
| 私有策略 | D0不变1024serial3；D1已179restore4096serial15。后续操作须鲜活actualidentity/allNPU16-noForeign/ownedidle/sourceSHA/CAS/nativeSELECTED/fallbackFalse/finallyrestore |

每域nnodes1/local16/world16，无EP/noKVconnector/AllGather；K3/KV3GiB/Graph4,8,16,32max32/seq8/HCCL768/W8A8/nativeMLA/nativeAsync/MTP3/noSP/noDSACP/overlapFalse/AtomicMQWorker。PP empty_token CPU metadata guard仅num_new_tokens==1 and new_token_ids时[-1]，不改算子/tensorpadding/MTP提交。nativebase993f2c.../methodfixed3e8ccc.../16markers每域真实证据175168；实际空分支触发及166次生copy因果unknown。

physicalauthority为Run175/restored及179根目录standalone roots/members/native resident/config，最新publicproof jobs/AUDIT-RUN179V3-20261003T1342Z/public_service_proof.json。public175精确owned SDKfinal0/observerterminal，仅public切换，modelsepochs同175168。resp_glm_run175_D0_new和resp_glm_run168_D1_new原GET/nativewire保留；179各parent续写16tokenchild/retrieve有效。旧D0 139/旧D1 139149156164 IDs GET/previous503beforeleaseRPC，无STORE复制/跨owner回放。

Mac仅SSH166：ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166；167通过166HOST ssh -o BatchMode=yes root@172.16.10.167，container166无SSH。repo /data/tiankuan/wio/Inference-Foundry；site /data/tiankuan/wio/glm52-pd/deploy；container glm52-single。只操作本任务资源，必要启动清理重启已授权。.controller.lock/.formal-test.lock/controller-owner.json，TTL30/heartbeat5/watchdog2。实际Zcode /usr/local/bin/zcode0.16.5，在site/private/zcode-relay-work由scripts/zcode_bridge.py run ABSjob.json，真实zcode --prompt→Result→bridge；模拟不替代。GPT关键研究代码裁决/Zcode执行监控归约/raw在SERVER。无browser/subagent/automation。

## 最近证据与当前机制

176177同all4input152/output4096/c4/countfallback/两域各2/两warm/t1024：TTFTP504000.8/4021.7ms严格FAIL，TPOT5/6；nativeprefill~3.31s，MTP.940/.924→.549/.634，TPS226.20→159.91不可全归因阈值。178两域t512prefill3.745/3.772s/TTFTP504329.5FAIL，conditionalREJECT，restoreD0serial3/D1serial13；非全域上界。

Run179 VALID V12 HTTPnative CPUtokenmemo/functionalmigration; aggregateSLOREJECT: cold6miss TTFTP504101.4ms FAILonly/TPOTP509.5-P9012.5; second4hit2miss TTFTP503867.8ms/TPOTP508.5-P908.8 all6PASS. Eachall4input152/output4096/c4/countfallback actual2each+2warm/gen16386/query475208/cache292864/6success/nativewire/SDK0; totalfunctional76+32772=32848/26complete/nohelperpreemptuncredited/source113/native32sameownedidle. Models175D0+168D1 epochs unchanged; public175exactowned retire SDKfinal0/observerterminal; public179HOST3171974/start277979053/V12/state125/SDKinit0active/newobserver. Correct4/42/GLM_OK_731 twice eachdomain/12semanticChat/nativepromptIDs/misses6hits6; bothSTOREGET rawsame/previous16-childretrieve/retiredIDs503preleaseRPC/no replication. D0same8192t1024serial3/actualSELECTED3reuse; D1ownedidleCAS4096serial13→t1024serial14/SELECTED14→restore4096serial15/fullpeer+hints. CPUmiss/tokenize andlargernativebody costinsideHTTP, no preloadedIDs; originalpublicbodySHA differsfromtransformednativebodySHA byappendedkv.prompt_token_ids, originalmemberbytes/sampling/nativewire unchanged. CPU2real10tokenize/32counter0/SDK0 plus11mockcontracts separated. Cold/reusefullCLI90.438703s/181.161377TPS and71.916619s/227.819386TPS; nativeprefill3.309/3.316 and3.317/3.326s, MTP.668/.832→.982/.951, noisolatedTPSgain. Memo14entry3981606B/32MiB/pending0/errors0; epoch+origin binding causedroutingmisses, notallhit. Audit65139B SHA242822259e3b113d49f1764f65a2c2c8233aad1df921a8c5f44474ac0e59bb19. FailedCPU1 importpath/AUDIT179-V2 localnameerrors preserved; correctedindependentCPU/audits no inference replay. No stablecapacity/globalbound/robustN4tail/full61440/KEEP/Current. Nextsamecandidate full61440/all4/c4actual2each/newKV salt/no CPUpreload/allcost-counted/models-public179-STORE retained.

CPU2 jobs/TOKEN-MEMO-CPU2-20261003T1323Z真实各域/tokenize全4+两warm，10calls/73740-81932/历史CPU3 IDs比对/native32counter0/SDK0；11项mockcontract标simulation，reduction23050B SHA53dd034cbe3fd69d85135972115f42fec38f8def928b547dfbcd9f67e8d44a18。nativeonline_rendererSHA5a8500fc... sourcepin。

V12 runtime/native_chat_token_memo.py/response_affinity_gateway_v12.py/native_engines_service_entry_v12.py复用unchangedV11租约/取消/状态/quarantine/drain/nativePD/wire，只plain单usertextChat已识别字段添加kv.prompt_token_ids。origin+nativeepoch+完整Chat/templatekwargs键，原cache_salt/采样保留；32MiB serializedIDs+source metadata byteaccount、128pending、singleflight/取消关闭回收。未知字段/tools/multimodal/Responses/previous/PD/签名query原native路径。miss同selectednative/tokenize在HTTP内、hit完整IDs、body增长全计E2E，无外部预载。公开原bodySHA和实际nativeSHA不同，audit逐条验证originalmemberbytes/采样/CPU原responseIDs/nativeoutputwire；不称byteidentical-input。

## 机制复用与下一问题

完整功能依据接口/源码/原生参数/行为，139147149-151168175179已有Chat/Completion/typedResponses/tools/reasoning/usage/采样/logprobs/STOREprevious/取消/nativeerrors/drain-readd/epochaffinity/persistentfault/nativeidentity/openarrival证据，手工清单不缩小范围。customID并发race/STORE复制/实时native进度unknown，helper3未知ACK不计成功。

152两域PP2各2/all4×81932→61440完整245760/fullCLI316.55TPS，TTFTP506264.3FAIL/TPOTP5011.4-P9011.9PASS；旧PP2 DCPeffectiveblock1024/cache72704，当前PP4block512/cache73216/t1024不同，只复用工作包不继承界。146开放48短/offered409.6TPS/晚期397–415/backlog14–16/TPOTP5035.74FAIL，不是稳定容量。147148变长12/coarseWorkSeconds/减预算有短延迟/TTFT取舍，无全SLOKEEP。162预计算CPUtokens计时外，不能叫普通text免费收益；179在线冷热计数真实，epochbound跨域共享未实现。

CACHE-MTP-CPU hiddenstatebackoff不可去掉；PP4kernel128/DCP4effective512/LCP73738→73216条件预测，actualcounter权威。PCP2原生不支持/PP2PDdecode不支持/HF22,42,62合法/budget8192inflight4；CPU接受不等于NPU fit。nativeiteration_tokens_total是frontend computedprefill+gen，不是SchedulerOutput/GPUstep；queue不含frontend/coreinbox，TTFT−prefill不是纯GPU/tokenizer界。105CPUprof不是硬件界，32EP32OOM/79DSACP-SPmathfail只原配置，不盲重放。

下一高信息实验同native175168/现public179 V12/c4/countfallback实际2each/两warm/freshKV盐/noCPU预载/output61440完整E2E；保留models/STORE/state125，D1baseline4096serial15→1024serial16→restore4096serial17，D0serial3不变。观察长上下文TPOT/MTP/KV/全部计数，不固定额外步骤/扫参/N4尾提升Current。

## Git与记录

Git权威branch glm5-3-autonomous-20261001，已同步HEADa891090ce093669abc254f5d1b0d4e859714525e/treeddb64cad055069f362628437b1bd7f15554486f8。checkpoint133原681记录175176终态/177ACTIVE，childa891记录bridgeUnicode-normalizedjob（JSON字段同/字节不同），全260ownedbyte/tree/commit/branch/index/origin严格CAS/no force/无关worktree保留。下一134记录177178179终态+CPU/V12源码与后续实际状态。大SSE/token数组原raw在SERVER只存refs/hash到Git；活动JSON固定快照。原manifest/source/raw不覆盖，GPToverlay manifest_record_v2/checkpoint_terminal_state/reduction_brief/summary关联code+Run+结论。points.jsonl/point/source_notes/source_evidence统一。VALID≠KEEP/running≠completed，next_check_at不是已有定时任务。
