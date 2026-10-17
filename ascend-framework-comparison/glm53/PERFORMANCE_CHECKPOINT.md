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
