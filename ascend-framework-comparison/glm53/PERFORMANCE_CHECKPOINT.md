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
