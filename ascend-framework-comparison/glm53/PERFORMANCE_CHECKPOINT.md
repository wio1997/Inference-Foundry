# 当前：C32 对照与 D 诊断已验收，候选待控制通道恢复

只做普通 W8A8 GLM-5.3 的 C32 对比与优化；C16/C64/L05保留历史，不新增测试。

| C32 部署 | 正式成功 | 全程输出 tokens/s | TTFT P50 秒 | TPOT P50 毫秒 | 固定 SLA |
| --- | ---: | ---: | ---: | ---: | --- |
| vLLM 参照 | 128/128 | 530.8125 | 21.1620 | 19.1527 | Fail |
| 当前 SGLang L04 对照 | 128/128 | 155.9674 | 14.3819 | 138.4531 | Fail |

当前 SGLang 32 个独立前缀的55776组件键/384次真实读回、全128请求计数、32个P原生缓存读取worker、32个D实际SDMA接收设备均独立核验通过。完整时钟492.4106秒，76800个正式D输出token；预热不计。两框架吞吐比3.4034倍。这是部署栈比较，包含图模式和实际缓存行为等差异；未证明纯框架因果。相同L04旧窗口165.6284TPS与当前相差−5.83%，不是统计噪声界限或新优化回归；小幅候选收益需匹配重复。

当前C32两步原生profile的32rank证据完整：统计中位占比计算9.073%、不可重叠通信52.758%、Free38.198%；每rank480次alltoallv、约556次流同步。采样只有两步，不能当作全程因果预算。自动停止的32个原生结束记录均核验通过；重复停止接口HTTP500/passFalse原样保留，以独立补充证据说明，不改原始确认状态。CPU43交付边界中位数不足1毫秒；内部D缓冲不在该测量范围。诊断不产生优化收益。

L07只选择D原生Ops派发/合并组件；文档和现场源码支持继续做实际兼容性验证。所需进程级通信缓冲配置1024MiB，须守住每DP至少161280个KV token；图禁用、MTP5、权重/KV精度、并行、P配置、MemCache与SDMA不变。27文件候选和26文件L04回退包已暂存四节点，两个SDK配置CPU预检通过；候选尚未启动或验收，没有收益结论。

部署前停止脚本在只读节点映射阶段退出，未停止模型/贡献进程/Meta。2069两次只读SSH均在61.241.77.34:60011握手超时、rc255；等待可用接入信息。原2029服务未被这些操作改动，断联后的现场健康尚不能重新确认。完整恢复点见RECOVERY_LATEST.md。

MemCache已先核官方版本对应部署文档：Meta→独立贡献进程READY→模型客户端dram=0GB；元数据查询与真实payload读回分别验收。文档模式不代替具体SGLang/GLM-5.3运行资格。汇总与来源见results/memcache-docs-C32-postrun-reconciliation-v3.json。

当前公开聚合摘要results/sglang-C32-L04-control-v3-public-summary-v1.json；原证书、请求ID、输入/输出、原始日志及trace保持原地，不纳入本次Git快照。

<!-- C32_ONLY_CURRENT_CHECKPOINT_END -->

## 以下为历史检查点

# GLM-5.3 ordinary W8A8: current accepted deployment measurements

Five conditions are independently valid; optimized SGLang C64 protection failed. All accepted conditions fail the fixed latency SLA. Primary throughput uses the full first-dispatch-to-last-terminal clock and only successful formal D output.

| Framework | C | Requests | Output tokens/s | TTFT P50 (s) | TPOT P50 (ms) | SLA |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vllm | 16 | 64/64 | 441.2158 | 6.5150 | 18.3460 | Fail |
| sglang | 16 | 64/64 | 103.0997 | 8.3098 | 124.1661 | Fail |
| vllm | 32 | 128/128 | 530.8125 | 21.1620 | 19.1527 | Fail |
| sglang | 32 | 128/128 | 165.6284 | 13.1921 | 136.4805 | Fail |
| vllm | 64 | 256/256 | 550.2465 | 53.0458 | 20.0138 | Fail |

SGLang C16 before/after:94.1764→103.0997tokens/s, observed+9.475percent in one matched pair. Repeat noise has not been estimated; no generalized or statistical gain claim. No unoptimized SGLang C32/C64 baseline exists. This compares native deployments, including different hardware groups/graph/cache/staging/SDK packaging; it is not pure framework attribution.

vLLM C64 original strict native V2 audit rejected four periodic GD_TO_LD tails (GOODdelta636, fourlastON_FLY1). Those raw values and the failure remain unchanged. The independent supplement verifies each of32Dworkers has20balanced WAIT andREAD_SIZE completions matching successful native and typed-model-state reads:640total,320logicalrequests×2workers. Every other V2 request/cache/bytes/graph/MTP/metrics/source/error/idle check passes. Official v1.2.1 ptracer source only emits a periodic point when BEGIN increases; an end-only period can leave its prior row visible. Installed SDK binary equivalence is not claimed; acceptance rests on the independent actual completion evidence. No SDK flush/change or model rerun was used.

Native vLLM D local-cache source fraction including warmup/formal is75.29percent atC16/C32 and2.60percent atC64; descriptor bytes are0.3016/0.6033/4.7530TB. These are observed native accounting and descriptors, not physical-wire bytes or proof of a specific eviction cause. The C16 PD-volume difference must not be generalized toC64. Exact counts/scope:results/vllm-native-cache-composition-C16-C32-C64-v1.json.

Current comparison:results/L04-matched-deployment-comparison-v2.json. vLLM C64 certificate:results/vllm-formal-C64-v2-measurement-v1.json,SHAfc6f065f561b5326fc910d8284723a9ec960fb33c4da89ce8837a575f5f06774. Supplement:results/vllm-C64-SDK-completion-review-v1.json; original rejection:results/vllm-formal-C64-v2-native-audit-attempt-v1.json.

SGLang C64 run2000 is invalid:210/256formal requests completed with correct counts;46failed. Initial64warmups/all-key payload gate passed, but35completed requests failed raw cache checks (26missingwholeobjects,8numericpartialhits,1optionalDstorageomission). All46failed requests join to native D300sWaitingForInput timeouts;19also have native P300sBootstrapping timeouts. The warm+formal Meta interval shows159537evictions/7816get_not_found, with0allocfail; successful submitted SDMA sends completed with native status0. These facts do not establish the complete causal chain or an L04 regression. Partial-run throughput is diagnostic and excluded from this table. Evidence:results/sglang-C64-native-failure-diagnosis-v1.json.

<!-- CURRENT_MATRIX_END -->

## Earlier checkpoints retained for provenance
# GLM-5.3 ordinary W8A8: measured L04 routing metadata trial

One full C16 before/after observation is valid after independent client and native audits. The D-only overlay batches three routing-count downloads into one and supplies the known repeat output size, preserving the native alltoall-normal algorithm, ordinary W8A8, MTP5/6, SDMA, graph/overlap settings, cache policy and hardware budget. CPU property/spawn fixtures, all32actualDworkers, all16D-DP short cases, two80K semantic cases and native cache/SDMA checks passed before the trial.

| C16 deployment | Successful formal requests | Full clock seconds | D output tokens/s | TTFT P50/P90/P99 seconds | TPOT P50/P90 ms | SLA |
| --- | ---: | ---: | ---: | --- | --- | --- |
| vLLM uniform v2 | 64/64 | 87.0322 | 441.2158 | 6.5150 / 12.7804 / 15.4191 | 18.3460 / 21.1469 | Fail |
| SGLang baseline | 64/64 | 407.7456 | 94.1764 | 8.7221 / 15.4866 / 24.2553 | 132.9702 / 148.9973 | Fail |
| SGLang L04 | 64/64 | 372.4549 | 103.0997 | 8.3098 / 16.6998 / 24.1512 | 124.1661 / 139.0624 | Fail |

Observed SGLang throughput is9.475%higher; TPOT median is6.621%lower. TTFT P90 increased and TPOT P99 is essentially unchanged. vLLM/SGLang throughput ratio moves from4.6850 to4.2795. This is one matched before/after observation; repeated full-run noise has not been estimated, and no statistical confidence interval or general workload guarantee is claimed. Matched request/prefix summaries are descriptive because closed-loop requests share resources. Whole-output bitwise equality is not required; the separate strict80K fact retrieval gates passed.

Both SG intervals have exactly1,404,182,052,864native PD descriptor bytes across warm+formal phases. These are not physical-wire bytes. MTP verification steps are9115before/9148after (5drafts perverify), with29358/29332accepted drafts. All32Ddevices receive typed state, all32Pworkers show native model cache reads in the formal interval, no new native errors appear after the preserved startup control-port diagnostics, and998original metric bodies/499complete16-DP pairs are independently rehashed/reparsed (sampled peak16). The unchanged descriptor volume and nearly unchanged MTP work are recorded as evidence, without claiming they explain every timing difference.

The raw L04 client report remains baseline_accepted=False because19Dstorage fields are omitted; the installed formatter/source-qualified supplement retains those fields as null and verifies all64P/Dprefixes. It does not fill raw values. All16real74400/1warmups,27888known component keys and192repeatable payload reads pass. Only38400formal D tokens count over the full firstdispatch-to-lastterminal372.45494829seconds. Native AISBench steady135.0683tokens/s is a separate48request window and is not the primary103.0997figure.

SGLang1954 remains running with L04. C32 is independently valid:128/128requests,76800formal output/full463.68872027seconds=165.628355tokens/s; matchedvLLMC32 is530.812539tokens/s (ratio3.2048). C32 TTFT P50/P90/P99=13.1921/39.8760/53.9882seconds; TPOT=136.4805/161.3670/199.1930ms, SLA failed. All32warmups,55776component keys/384payloadreads, nativeSDMA/all32Ddevices/modelSDKreads/completeDmetrics pass; original client accepted all128cache objects. No unoptimizedSG32baseline exists, so no C32 optimization delta is claimed. Evidence:results/sglang-L04-C32-v1-measurement-v1.json,SHAdb62ac7c12a1154933203ff42435a99e7ba6a85d0c4ba2f48d7c00ca1ed64610. vLLMC64uniformv2 is active; its oldv1two-P-disconnect run remains invalid. SG64 follows actualmatchingvLLMC64. No profile, cache flush, model restart, SDK change or graph/precision change is part of the measured L04 trial.

The corresponding public installed-version argument hook rejects D radix caching with NEXTN (onlyNone/DSPARKare accepted); actualinstalleddecode.py matches that commit byte-for-byte, but the installedargumenthook has not been separately captured. The eligibility review is conservative and no flag was enabled or validator bypassed. Exact source/scope: `results/sglang-decode-radix-eligibility-review-v1.json`.

Evidence: `results/sglang-L04-C16-v1-measurement-v1.json` (SHA8208130174e25df5a669a265d222b74a3fdb313f7ee809f278d52b22bbed5a43), `results/sglang-L04-C16-v1-native-v1.json`, `results/sglang-L04-C16-observed-before-after-v1.json`, `results/sglang-metadata-native-readiness-v1.json`. Source/config/compactproofs are published; newerL04results/fixes await the next selective checkpoint. Raw IDs/prompts/outputs/logs/traces/weights/SDKbinaries remain excluded.

## Earlier accepted baseline and diagnostic setup (historical)

# GLM-5.3 ordinary W8A8: accepted C16 deployment gap and SGLang optimization

The matched C16 comparison is accepted after independent input/count/lifecycle/cache/SDMA/native-metrics checks. Both deployments fail the fixed latency SLA. Main objective is to reduce SGLang's measured gap with bounded profiling and a single-mechanism C16 A/B; remaining C32/C64 cases do not block that first optimization. No optimization gain is established.

| Stack, C16 | Formal requests | Full clock seconds | D output tokens/s | TTFT P50/P90/P99 seconds | TPOT P50/P90 milliseconds | SLA |
| --- | ---: | ---: | ---: | --- | --- | --- |
| vLLM v2 | 64/64 | 87.0322 | 441.2158 | 6.5150 / 12.7804 / 15.4191 | 18.3460 / 21.1469 | Fail |
| SGLang v2 | 64/64 | 407.7456 | 94.1764 | 8.7221 / 15.4866 / 24.2553 | 132.9702 / 148.9973 | Fail |

Observed deployment ratios: vLLM/SGLang output throughput4.6850; SGLang/vLLM TTFT median1.3388 and TPOT median7.2479. These are end-to-end observations, not pure framework causal attribution or kernel timings. Current stacks each have64die/4TiB HBM, same ordinary W8A8/model/tokenizer/input IDs and fresh-connection AIS client v2 without automatic retries. Each independent74400prefix has one real74400/1warmup, all-key/repeated-payload readback gate and four distinct5600suffixes. Formal requests are80000/600; full first-dispatch-to-last-terminal clock includes all64 requests and all gateway/queue/transfer time. P/warmup output is excluded. Natural HBM residency is preserved.

SGLang's frozen client originally rejected25Dcache objects lacking `storage` (39/64 strictcachechecks). Original report SHA cb26620a032d48f3933017598ed600d8fe786de9dde87c80e43952c4bbf93890, baseline_accepted=False and all missing fields are preserved. Actual installed output_streamer emits the D storage field only for positive inherited P storage or enabled local HiCache; D HiCache is disabled. All25omitted objects have74368device/0host, with matching P74368device/0host/0storage. A pure formatter AST fixture reproduces all64native P/D objects; independent supplement verifies prefix hits while retaining25Dstorage nulls. Seven negative cases are rejected. No field is filled, report changed, model request repeated or SDK patched. Native postcapture also verifies actual model MemCache reads, SDMA typed-state transfers to all32Ddevices,5drafts/verify for all64requests, no new native errors, and all1110original D metric body hashes/555complete16-DP pairs; sampled D running peak16. SG has18owned NPU PID contexts pernode: provider+modelparent+16registeredworkers, verified by exact identities; vLLM's17-row constant is inapplicable.

| Actual configuration | vLLM | SGLang |
| --- | --- | --- |
| Attention TP / DP, P | 8 / 4 | 8 effective / 4 (globalTP32) |
| Attention TP / DP, D | 2 / 16 | 2 effective / 16 (globalTP32) |
| EP | 32 | 32 |
| D MTP | 5proposed/verify, nativeacceptance audited | 5proposed/verify;45575proposed/29358correct/9115verify |
| Target D graph | Native FULL_DECODE_ONLY runtime execution | decodegraph disabled; eager |
| Schedule overlap | NativevLLM implementation | explicitly disabled |
| KV tokens/attentionDP P / D | 299136 / 229248 | 254848 / 181632 |
| External pool | 568GiB | 568GiB plus268.397GiB nativePstaging |
| D native radix reuse | Actual74240local+5760incoming formal | disaggregationdecode_radix=False, DHiCache=False |
| Request logs / distributed traces | Nativeaudit overhead disclosed | log_requests=False, enable_trace=False; metrics=True |

SGLang native eager/overlap-disabled settings are the measured diagnostic starting configuration, not a final optimized service. Compatibility and reasons for enabling/disabling graph, overlap and alternative communication paths must be established from the exact installed source, official guidance and real traces. Graph difference is a candidate mechanism, not an established4.685xexplanation. Native D storage accounting is inherited P cache provenance, not direct D L3 I/O. Raw SDMA descriptor bytes are not measured physical-wire traffic. Different cache placement/commonmax, SDK packaging, extra SG staging and CPU43 collocation remain attribution limits. One current accepted run per stack establishes this gap; repeat noise is not yet quantified.

Read-only installed profiler control/source review and seven CPU control fixtures are complete. SG1920 is the next diagnostic: unchanged C16 client/cohort, D-only native CPU/NPU8forwardsteps after all16DPrunning admission, explicit noStack/noShapes, wallguard and stop-in-finally, all32worker trace/stop verification required. Profiled timing cannot replace this unprofiled baseline. Raw traces remain node-side; later optimization requires a controlled unprofiled C16before/after with rollback. C32/C64 protection cases remain queued.

Accepted vLLM C32v2 is128/128,144.68384672s/530.81253879tokens/s/SLAfailed. Historical v1C16/C32 remain452.5986/512.9852tokens/s, separate from currentv2. The first C64v1 attempt had254/256strictsuccess and two P ServerDisconnectedError before headers/noDdispatch: complete282.36720022sclock and539.7227successful-Dtokens/s preserved, baseline INVALID. CPU loopback reproduced reused-POST-close failure and qualified fresh connections, supporting v2 without proving exact original peer-close cause. No SDK/system change.

Compact evidence: `results/C16-paired-deployment-gap-v1.json`, `results/sglang-formal-C16-v2-measurement-v1.json`, `results/sglang-formal-C16-v2-native-v1.json`, `results/sglang-formal-C16-v2-cache-annotation-review-v1.json`, `results/vllm-formal-C16-v2-measurement-v1.json`, `results/vllm-formal-C32-v2-measurement-v1.json`. Raw input token IDs, prompts, outputs, logs, traces, weights and SDK binaries are excluded from Git.
