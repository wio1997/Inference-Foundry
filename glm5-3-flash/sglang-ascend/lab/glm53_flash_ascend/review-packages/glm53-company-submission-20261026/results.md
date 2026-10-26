# Compact R2–R6 results

This is a public numeric summary, not a mirror of the private original evidence. Source publication: [`1505f60337e7e310bdfaffbf649b9ee76b1d3fc7`](https://github.com/yanceng305-collab/sglang-glm53-flash-ascend-lab/commit/1505f60337e7e310bdfaffbf649b9ee76b1d3fc7) (separate access permission required). No new Ascend execution was performed to create this document.

All26 canonical performance services are listed below; the two preliminary performance services are retained separately. Values are copied from fixed original summaries. p50/p90/p99 are never averaged; milliseconds below are display-only conversions from original seconds. `UNKNOWN` means the selected source did not establish the field.

| Round / service | Formal success/planned | Warmup success/planned | TTFT p50/p90/p99 s | TPOT proxy p50/p90/p99 ms | E2E p50/p90/p99 s | Output token/s |
|---|---:|---:|---|---|---|---:|
| R2 baseline-c1 | 24/24 | 4/4 | 0.972 / 1.084 / 1.100 | 457.279 / 467.791 / 470.933 | 59.020 / 60.490 / 60.851 | 2.175523 |
| R2 baseline-c4 | 24/24 | 4/4 | 1.547 / 1.686 / 1.719 | 498.473 / 507.552 / 510.967 | 64.657 / 66.041 / 66.042 | 7.942994 |
| R3 a-c1-04 | 8/8 | 1/1 | 6.640 / 6.661 / 6.686 | 400.939 / 406.289 / 408.570 | 416.789 / 422.279 / 424.606 | 2.450188 |
| R3 b-c1-04 | 8/8 | 1/1 | 5.906 / 5.946 / 5.970 | 243.611 / 244.675 / 244.797 | 255.109 / 256.204 / 256.316 | 4.027260 |
| R3 a-c4-04 | 8/8 | 4/4 | 16.105 / 25.023 / 25.034 | 428.494 / 437.016 / 438.409 | 454.481 / 455.623 / 455.623 | 9.012282 |
| R3 b-c4-04 | 8/8 | 4/4 | 14.497 / 22.318 / 22.344 | 250.524 / 258.177 / 259.378 | 270.783 / 271.764 / 271.765 | 15.125544 |
| R4 a-c1-01 | 8/8 | 1/1 | 5.878 / 5.900 / 5.912 | 239.149 / 240.765 / 240.867 | 250.509 / 252.161 / 252.316 | 4.089172 |
| R4 a-c4-01 | 8/8 | 4/4 | 14.466 / 22.286 / 22.329 | 245.149 / 252.393 / 254.829 | 265.254 / 267.221 / 267.222 | 15.440201 |
| R4 b-c1-01 | 8/8 | 1/1 | 5.746 / 5.773 / 5.777 | 235.209 / 237.184 / 237.699 | 246.383 / 248.335 / 248.899 | 4.161365 |
| R4 b-c4-01 | 8/8 | 4/4 | 14.093 / 21.723 / 21.744 | 244.612 / 251.849 / 253.672 | 264.331 / 265.811 / 265.811 | 15.494324 |
| R5 a-c1-01 | 8/8 | 1/1 | 5.728 / 5.792 / 5.795 | 237.245 / 240.021 / 240.109 | 248.463 / 251.293 / 251.420 | 4.118477 |
| R5 a-c4-01 | 8/8 | 4/4 | 14.084 / 21.752 / 21.761 | 246.342 / 253.907 / 254.665 | 266.091 / 266.703 / 266.704 | 15.392590 |
| R5 b-c1-01 | 8/8 | 1/1 | 5.673 / 5.712 / 5.737 | 235.414 / 239.454 / 239.548 | 246.504 / 250.691 / 250.761 | 4.146345 |
| R5 b-c4-01 | 8/8 | 4/4 | 14.028 / 21.605 / 21.609 | 244.763 / 252.128 / 253.427 | 264.425 / 265.478 / 265.478 | 15.489148 |
| R5 b-c1-02 | 8/8 | 1/1 | 5.694 / 5.722 / 5.723 | 234.276 / 235.896 / 236.949 | 245.353 / 247.025 / 248.070 | 4.191735 |
| R5 b-c4-02 | 8/8 | 4/4 | 14.006 / 21.605 / 21.630 | 240.271 / 247.787 / 248.610 | 259.804 / 260.454 / 260.455 | 15.765193 |
| R5 a-c1-02 | 8/8 | 1/1 | 5.731 / 5.774 / 5.780 | 236.474 / 238.625 / 239.138 | 247.653 / 249.815 / 250.362 | 4.128714 |
| R5 a-c4-02 | 8/8 | 4/4 | 14.080 / 21.732 / 21.742 | 246.982 / 254.766 / 254.822 | 266.737 / 266.793 / 266.794 | 15.354800 |
| R6 a-c1-01 | 8/8 | 1/1 | 5.684 / 5.730 / 5.747 | 237.149 / 238.641 / 240.351 | 248.329 / 249.843 / 251.571 | 4.124432 |
| R6 b-c1-01 | 8/8 | 1/1 | 5.560 / 5.569 / 5.571 | 236.536 / 240.271 / 240.491 | 247.526 / 251.356 / 251.583 | 4.132174 |
| R6 b-c1-02 | 8/8 | 1/1 | 5.572 / 5.600 / 5.603 | 233.040 / 234.028 / 234.610 | 243.961 / 244.966 / 245.568 | 4.219361 |
| R6 a-c1-02 | 8/8 | 1/1 | 5.685 / 5.711 / 5.739 | 239.537 / 241.910 / 243.296 | 250.724 / 253.146 / 254.628 | 4.075516 |
| R6 a-c4-01 | 8/8 | 4/4 | 14.038 / 21.623 / 21.634 | 243.275 / 250.209 / 252.864 | 262.900 / 265.065 / 265.065 | 15.578311 |
| R6 b-c4-01 | 8/8 | 4/4 | 14.265 / 21.034 / 21.051 | 245.430 / 252.126 / 254.888 | 265.340 / 267.586 / 267.587 | 15.435498 |
| R6 b-c4-02 | 8/8 | 4/4 | 14.248 / 21.032 / 21.047 | 240.984 / 248.392 / 248.911 | 260.776 / 261.198 / 261.198 | 15.705709 |
| R6 a-c4-02 | 8/8 | 4/4 | 14.053 / 21.690 / 21.694 | 245.216 / 252.141 / 254.890 | 264.906 / 267.139 / 267.139 | 15.460556 |

## Preliminary, failed and diagnostic records

The JSON retains every one of47 service/attempt records, including limited quality and profile diagnostics. Diagnostics are not mixed into the unprofiled performance table.

- R3 a-quality-03: actual server seed 659417781; 6/6 successful, TTFT p50/p90/p99 16.244 / 33.140 / 39.754 s, TPOT 531.228 / 619.595 / 636.780 ms, E2E 104.870 / 116.269 / 116.808 s, output 3.123466 token/s. Excluded from canonical04, not relabelled as numeric failure.
- R3 a-c1-02: actual server seed 210345776; 8/8 successful, TTFT p50/p90/p99 6.684 / 6.764 / 6.783 s, TPOT 405.257 / 408.110 / 410.351 ms, E2E 421.341 / 424.171 / 426.441 s, output 2.426212 token/s. Excluded from canonical04, not relabelled as numeric failure.
- R3 a-c4-02: actual server seed 947671466; 8/8 successful, TTFT p50/p90/p99 16.105 / 24.952 / 24.961 s, TPOT 413.536 / 422.008 / 425.766 ms, E2E 439.153 / 442.937 / 442.937 s, output 9.326699 token/s. Excluded from canonical04, not relabelled as numeric failure.

## What changed and what it supports

- R2: paired Flash implementation and compatibility baseline, Torch mHC. 512/128 and differing GC/observer conditions prevent a fair direct R2-to-later speedup claim.
- R3: only the FP32 HC4 Sinkhorn normalization tail was fused, preserving20iterations/epsilon placement and surrounding mixing. Canonical reported throughput increased64.365359%/67.832562% for C1/C4. Selected ordered TileLang AOT passed96/96; initial failures and frontend ABI problems remain recorded.
- R4: forward-local KPool CPU metadata snapshot reuse. Target readback calls normalized to11WK calls went44→4, not a90.91% model gain. C1 TPS+1.765464%; C4+.350532% fell within block variation.
- R5: forward-local NoPE KV-length device conversion reuse. Two independent-service equal-weight TPS means changed+1.1020%/+1.6488%; all four pairs had lower TTFT/TPOT/E2E, but magnitude varied and no statistical-significance claim is made.
- R6: same engine,2048/4096 prefill-budget comparison. Keep2048:4096 helps C1 TTFT, trades off C4 central/tail latency, adds sampled memory and has no uniform decode/throughput gain.

R3/R4 blocks are not independent service restarts. R5/R6 each contain two independent services per configuration/concurrency. None of these improvements is multiplied into a project cumulative speedup. Runtime candidates remain opt-in/experimental; source availability is not production qualification.

## Failures, selection and limits

- **R2 scope_and_observer_limit**: Complete model baseline established; no optimization A/B. Automatic freeze_gc failed for C1/C4 but succeeded for separate profiling; C4 native identity sampling overlapped early formal timing. Do not present as bare production speed.
- **R3 numeric_failures_retained**: Initial Triton and public TileLang tail each passed94/96; initial downstream BF16 T128 error reached10ULP and T2048 reached13040ULP. Ordered reductions then passed96/96 without relaxing atol2e-7/rtol2e-6/maxrel2e-5 and BF16<=1ULP. The first Triton harness used a copied reference, while the later harness exercised actual False/True pre paths; these two component runs are not a kernel-only A/B. Ordered Triton remains an alternative, not the full-model B.
- **R3 ABI_and_AOT_selection**: Release wheel four ELFs required GLIBC2.38 while target was2.35; no libc upgrade. Isolated frontend/CANN coexistence5cases had2success/3failure. Selected B uses ctypes AOT and current stream, not importing private TileLang/TVM in the model process. AOT probe01 refused before96cases at Git-clean guard; dirty cause UNKNOWN. Probe02 and full-model canonical04 use ed216.
- **R3 control_and_seed**: First A failed before requests in ModelSlim/CANN format_cast; second A was refused by occupancy preflight before a model child. Preliminary success used differing service seeds and is not canonical04. Old B01/02/03 plans were not executed. R3 profile auto-stop/export succeeded; the final HTTP500 was a repeated STOP, not capture failure. Preserve original protocol UNKNOWN separately from the correction.
- **R4 mechanism_not_total_speedup**: Target KPool readback sets per11WK calls changed44copies to4 (-90.91% target calls), not -90.91% all copies or latency. C4 two blocks had opposite throughput directions. Full native-map closure remained NOT_ESTABLISHED after4416-group ValueError.
- **R5 limited_attribution**: Copy/allocation dispatch per normalized CPU-step decreased about10, but stacks were disabled and exact KV/query site counts remain UNKNOWN. Formal four-pair directions improved; magnitude varied. Known native subsets do not close every mapped library or ABA/native mutation behavior.
- **R6 mixed_tradeoff_and_control_recovery**: Only2048/4096 prefill budgets differed. Keep2048 baseline: B C1 TTFT p50 improved about2%, C4 p50 regressed about1.4-1.6% while p99 improved about2.7-3.0%; decode/TPS varied and sampled HBM rose1120-1123reportedMB. Original B port preflight Errno98 occurred before prepare; committed continuation ran only original remaining9 points, with no model/request replay. Some control stderr was not retained; exact historical control cause is UNKNOWN. No new R6 profile.

- This public contribution contains code and compact numeric evidence, not a mirror of private raw evidence. Fixed private-source links require separate permission.
- No prompts, raw SSE, raw server logs, full environment, authentication information, machine addresses, weights, compiled libraries, caches or profiler traces are copied.
- R2 is 512/128 under its recorded diagnostic/GC/observer conditions. R3-R6 main A/B is 16384/1024. No cross-round cumulative acceleration or fair R2-to-R6 speedup is established.
- R3 canonical04 excludes successful preliminary services with different server seeds; their metrics remain separately listed.
- R4 has one service lifetime per configuration/concurrency with two within-service blocks. R5/R6 have two independent services per configuration/concurrency. These are descriptive experiments, not confidence intervals or production qualification.
- TPOT is first-to-last nonempty SSE receipt interval divided by output_tokens-1, not exact token ITL. HBM observations are periodic samples, not continuous peaks.
- Finite natural-stop checks are not full precision/logit/MoE-routing/long-state equivalence certification. Graph/MTP/PD, other hardware, long stability and full native/code-generation closure are not established.
- Full-device critical-path attribution cannot be independently reconstructed from this compact package. Host-self and multi-rank duration sums are not request wall-clock savings.
- Exact benchmark replay additionally requires the omitted controlled workload and environment artifacts; a fresh public workload is a new experiment and cannot inherit these results.

Exact per-service metrics, formal and warmup counts, token-frequency observations, blocks, code identities, component observations and selected source-file hashes are in [compact-evidence.json](compact-evidence.json). Original-run, intermediate-derived-stage and final fixed-private-Git hashes are separate domains. Original identities are recorded values copied only from explicit destination-path index mappings, otherwise UNKNOWN; their intermediate-stage bytes need not equal a later exported Git projection, and no original-byte reconstruction is claimed. The referenced private publication holds the full redacted original reports, responses and failure evidence; it is not reproduced by this public contribution.
