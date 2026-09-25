# 执行日志

- `2026-09-24T15:17:37Z` Loop 已冻结。下一步：Read borrowed target grouped-matmul implementation and trace 86 calls to exact layer, shape, dtype and backend; rank candidate substitutions by device bound.

- `2026-09-24T15:20:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run108`（design-check）。

- `2026-09-24T15:21:10Z` Run `run108` 记录为 `pass`；正确性为 `not-applicable`。Read-only source/trace audit passed: DeepSeek V4 Flash config has 43 hidden layers; all eight rank cycle64 target windows have exactly 43 GMM1 SwigluQuantWeightNzV2 and 43 GMM2 WeightNz kernels. Rank0 sums 6.425 and 3.646 ms, respectively. Source maps GMM1 to DeviceOperator.npu_grouped_matmul_swiglu_quant and GMM2 to npu_grouped_matmul_gmm2 through moe_mlp; actual tensor shapes and replacement parity are not yet measured.

- `2026-09-24T15:29:11Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run109`（design-check）。

- `2026-09-24T15:29:43Z` Run `run109` 记录为 `pass`；正确性为 `not-applicable`。Corrected Run108 GMM1 source attribution: frozen quant group_size=0 makes W4A8 weights per-channel; trace GMM1 uses moe_mlp.py custom grouped_matmul_swiglu_quant_v2 branch, while GMM2 uses DeviceOperator.npu_grouped_matmul_gmm2. Checkpoint packed I8 layer0 expert0 shapes are w1/w3 [1024,4096] and w2 [2048,2048]. Run108 43+43 kernel counts remain valid; live routed input shapes and faster variant remain unmeasured.

- `2026-09-24T16:10:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run110`（review）。

- `2026-09-24T16:15:25Z` Run `run110` 记录为 `invalid`；正确性为 `invalid`。Bounded Zcode environment check exited 0 after 188.845s and 27 provider requests, but headless build mode denied docker exec, npu-smi and HTTP access (No permission client configured for Bash). Configured model deepseek/deepseek-flash; actual model was not independently observed in output, so model_verified=false. HEAD 1b7c0e0 and 8 profile windows matched direct Sol recheck; Sol directly verified all eight NPUs idle and service stopped. Delegated result is partial and excluded from product evidence.

- `2026-09-24T16:56:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run111`（review）。

- `2026-09-24T16:57:50Z` Run `run111` 记录为 `pass`；正确性为 `not-applicable`。Minimal Zcode plan-mode ping succeeded: exit 0, 14.735 s, one provider request, exact sentinel response. Independent local model I/O record identifies request provider deepseek/model deepseek-flash and response modelId deepseek-flash. This validates short text connectivity only; Run110 tool permissions remain blocked and this is not product performance evidence.

- `2026-09-24T17:00:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run112`（review）。

- `2026-09-24T17:02:25Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run113`（review）。

- `2026-09-24T17:03:13Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run114`（review）。

- `2026-09-24T17:05:10Z` Run `run112` 记录为 `pass`；正确性为 `not-applicable`。Zcode DeepSeek build-mode minimal Bash pwd executed successfully (tool result /data/wio/Inference_Foundry, no error), proving Bash is not globally broken. Two provider requests, exit 0. Local model I/O confirms deepseek/deepseek-flash request and deepseek-flash response.

- `2026-09-24T17:05:10Z` Run `run113` 记录为 `pass`；正确性为 `not-applicable`。Isolated read-only direct Zcode yolo diagnostic: DeepSeek called Bash once for docker inspect and received exit 0/status running. This establishes the permission-mode contrast only; yolo was diagnostic, not adopted for delegation or product evidence.

- `2026-09-24T17:05:10Z` Run `run114` 记录为 `pass`；正确性为 `not-applicable`。Controlled Zcode build-mode docker inspect was blocked before execution with No permission client configured for Bash; CLI exit 0 only reports the agent response, not tool success. The contrast with Run113 confirms the headless build permission client gap. DeepSeek model identity verified from local model I/O.

- `2026-09-24T17:14:08Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run115`（profile）。

- `2026-09-24T17:33:02Z` Run `run115` 记录为 `pass`；正确性为 `not-applicable`。Eight-rank diagnostic shape capture completed under legal 12x1024 requests (12/12 success). Each TP/EP rank records the fixed 96-token target graph MoE envelope as int8 [576,4096], group_list int64 [32] count mode, 32 local experts, packed w1 int32 [32,4096,512], w2 int32 [32,2048,512]. Shapes identical across ranks; live per-expert counts are not captured. Temporary borrowed source restored byte-identically, service stopped. 544.725 diagnostic tok/s is not formal E2E.

- `2026-09-24T17:36:46Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run116`（design-check）。

- `2026-09-24T17:37:13Z` Run `run116` 记录为 `pass`；正确性为 `not-applicable`。Independent Sol recomputation over 15 valid synchronized Run107 windows: median GMM kernel sum 9.96625 ms, communication union outside compute overlap 10.2475 ms, and non-GMM compute-union lower bound 29.97925 ms. GMM remains bounded candidate, not proven highest value. Run115 provides static [576,4096]/32 expert shape envelope but no live counts. Requested Astra Medium second view had no independently observable model ID and is advisory only. No product benchmark or Runtime mutation.

- `2026-09-24T23:41:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run117`（profile）。

- `2026-09-24T23:44:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run118`（profile）。

- `2026-09-24T23:46:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run119`（design-check）。

- `2026-09-24T23:46:47Z` Run `run117` 记录为 `pass`；正确性为 `not-applicable`。Synchronized Run107 first reduce-scatter across eight ranks: cycle64 start skew20.533ms/end skew0.010ms, cycle65 start skew9.370ms/end skew0.017ms. Earlier ranks wait inside same collective, so measured communication union is not directly removable transfer time. Diagnostic synchronization may amplify skew.

- `2026-09-24T23:46:47Z` Run `run118` 记录为 `pass`；正确性为 `not-applicable`。Unsynchronized Run106 confirms first collective peer-wait signature: two profiled cycles start skew10.193/9.735ms and end skew0.0065/0.01175ms. Prepare_target entry already has9.637/10.510ms rank skew; cycle64 proposer-end skew10.466ms propagates to next prepare entry10.510ms. Profiled cycles only; do not generalize proposer duration to steady product.

- `2026-09-24T23:46:47Z` Run `run119` 记录为 `pass`；正确性为 `not-applicable`。Eight-rank steady Run98 cycles64-255 show target median46.575ms and proposer6.400ms, DSpark model5.987ms. Formal Run99 cohort wall median57.044ms/cycle is consistent. Thus profiled first-collective wait is not a demonstrated ~10ms intrinsic communication opportunity; prioritize bounded target GMM candidate over HCCL transfer tuning, while preserving whole-stage correctness and measurement gates.

- `2026-09-24T23:49:43Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run120`（profile）。

- `2026-09-25T00:07:15Z` Run `run120` 记录为 `invalid`；正确性为 `invalid`。Run120 invalid: request max_tokens=384 violated frozen Extreme serving exact max_tokens=1024 guard; engine exited, 11/12 success is unusable, no live counts. Temporary probe restored, 8 NPUs idle.

- `2026-09-25T00:08:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run121`（profile）。

- `2026-09-25T00:28:36Z` Run `run121` 记录为 `pass`；正确性为 `not-applicable`。Legal 12/12 1024-token requests; 16 snapshots across 8 ranks at cycles64/65. Exactly 43 live and 43 static refs; all 43 live ordinals change on every rank, cross-rank token count 576 per layer, active experts median 15/32 per rank-layer (688 samples). Diagnostic only, no E2E comparison.

- `2026-09-25T00:31:27Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run122`（design-check）。

- `2026-09-25T00:32:09Z` Run `run122` 记录为 `pass`；正确性为 `not-applicable`。Run107 15 synchronized windows: GMM1 median6.3785ms, GMM2 median3.58775ms, sum9.96625ms. Borrowed A8W4 GMM1 kernel reads live count and skips MatMul for zero-M experts, so Run121 sparse routing is already accounted for in GMM1. No >=5ms replacement established.

- `2026-09-25T00:34:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run123`（design-check）。

- `2026-09-25T00:37:09Z` Run `run123` 记录为 `pass`；正确性为 `not-applicable`。Run107 stable non-GMM target families across 15 synchronized windows: quant matmul4.8205ms, Compressor3.36575ms, HC pre2.961ms, scatter cache2.34275ms. These are sums, not removable critical-path benefits; no single family >5ms. Next bound GMM1 route sensitivity.
