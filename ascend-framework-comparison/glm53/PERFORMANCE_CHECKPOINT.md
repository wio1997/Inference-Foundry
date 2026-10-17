# GLM-5.3 ordinary W8A8: partial baseline checkpoint

This checkpoint accepts the vLLM C16 and C32 measurements after independent client and native audits. Both fail the fixed aggregate SLA. C64 is active; SGLang baseline series and profiling/optimization remain outstanding. No framework performance winner or optimization gain is established.

| vLLM client C | Formal requests | Full clock seconds | D output tokens/s | TTFT P50/P90/P99 seconds | TPOT P50/P90 milliseconds | Aggregate SLA |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 16 | 64 | 84.8434 | 452.5986 | 6.6536 / 12.8744 / 15.4275 | 18.2287 / 20.0913 | Fail |
| 32 | 128 | 149.7119 | 512.9852 | 21.2093 / 24.0138 / 35.8011 | 20.1644 / 22.2708 | Fail |

The denominator covers first formal dispatch through last terminal; output numerator counts only successful D output. Each request is exactly80000 input/600 output tokens. Each independent74400 prefix receives one real short warmup before four fresh5600 suffix variants; warmup/P-control outputs are excluded. Natural L1/L2 residency is preserved; no forced L3 load. AISBench3.1 native steady metrics are separate secondary windows.

All16/32 known prefix key sets and repeated first/last-page payload samples passed before the formal clock. Native postrun evidence includes real SDMA P→D transfers, original SDK interval counters, native model/cache events, current model identities, and all16 D attention-DP metrics. Sampled total D running peaks are15/16 respectively; client C32 does not imply32 simultaneous decodes. Native D cache accounting proves74240 local-HBM positions plus5760 incoming positions per formal request; API cached80000 alone cannot mean all80000 were locally resident.

The C32 node31 capture returned rc0 with incomplete JSON. It remains unchanged in run1804. A new read-only capture in1805 recovered the same owned runtime evidence through byte-count/SHA-verified gzip framing. C32 acceptance followed independent complete decoding and native auditing, without rerunning the workload. The transport failure cause is unknown.

P HTTP-leg median share of client TTFT is91.12% atC16 and96.48% atC32. This leg includes queue, cache preparation and compute, so this is not pure NPU prefill time or a proven optimization mechanism. Source-only profiler preparation exercised no native profiler and issued no model requests.

Sources: `results/vllm-formal-C16-measurement-v1.json`, `results/vllm-formal-C32-measurement-v1.json`, corresponding native-v2 and role-leg-timing-v1 reports; `BENCHMARK_CONTRACT.md`. Exact raw artifact SHA indexes remain in compact reports; raw prompt IDs/text/logs/SDK binaries are excluded from Git.

Both stacks use568GiB external pool, with different placement/common-max and SGLang268.397GiB native staging. Shared HBM hardware budgets and ordinary W8A8 identity are fixed. Future comparison is an end-to-end deployment-stack comparison until paired hardware/compatible-SDK controls establish narrower attribution.
