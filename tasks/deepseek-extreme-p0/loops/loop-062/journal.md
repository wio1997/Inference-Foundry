# 执行日志

- `2026-09-26T01:18:00Z` Loop 已冻结。下一步：Audit three exact Compressor shape source input/weight/workspace traffic and dependent scatter/attention path, then select the highest-value legal same-state intervention or pivot architecture

- `2026-09-26T01:19:12Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run256`（design-check）。

- `2026-09-26T01:20:06Z` Run `run256` 记录为 `pass`；正确性为 `not-applicable`。Run246 latest16 rank-cycle windows pass exact Compressor source/shape/dtype/count gates. Two BF16 weight tensors 0.608174GB/cycle, nominal two-projection 58.385GFLOP, reported AIC+AIV read1.530887GB and writes0.108488GB across62 calls. Unexplained read0.922713GB includes X/state/metadata/workspace/repeats and is not established waste. Source shows per-tile X and weight GM-to-L1, workspace Fixpipe and SyncAll dependency. Small c4 shape only1.145ms profiled total, so no standalone optimization promoted

- `2026-09-26T01:27:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run257`（design-check）。

- `2026-09-26T01:27:53Z` Run `run257` 记录为 `pass`；正确性为 `not-applicable`。Run246 8-rank graph trace finds two 256MiB BF16 TensorMove tasks near Target tail; source strongly implicates MTP pre-hc_head allgather and stash, but causal status awaits gated intervention/counter. No task-sum E2E claim.

- `2026-09-26T01:29:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run258`（profile）。

- `2026-09-26T01:40:39Z` Run `run258` 记录为 `invalid`；正确性为 `invalid`。Compiler rejected first opt-in patch: getenv in model forward caused TorchDynamo graph break. No benchmark or profiler result. Service stopped, 8 NPUs idle, borrowed deepseek_v4.py SHA restored exactly. Patch redesigned to constructor-fixed bool for Run260.

- `2026-09-26T01:40:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run260`（profile）。

- `2026-09-26T02:01:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run261`（profile）。

- `2026-09-26T02:06:22Z` Run `run260` 记录为 `pass`；正确性为 `pass`。Constructor-fixed DSpark-only MTP stash gate ran FULL Graph, 48 warmup+12 sampled requests all HTTP success, 16/16 rank/cohort runtime pass. Profiler run only; source restored exact SHA, 8 NPUs idle. Run261 exports counters; no formal TPS claim.

- `2026-09-26T02:06:23Z` Run `run261` 记录为 `pass`；正确性为 `not-applicable`。16/16 latest rank-cycle FULL Graph windows valid. Both 256MiB BF16 copies absent vs baseline2/cycle; median other read -0.552GB and write -0.543GB, HCCL 264 vs265 in 15/16 windows; one asynchronous spillover rank-cycle 267. Synchronized Target scope median54.796->54.418ms, diagnostic only; graph task savings not additive E2E.

- `2026-09-26T02:06:23Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run259`（benchmark）。

- `2026-09-26T02:08:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run262`（design-check）。

- `2026-09-26T02:08:04Z` Run `run262` 记录为 `pass`；正确性为 `not-applicable`。HCCL trace payload signature loses one 393216B AllGather per rank-cycle in 15/16 latest windows, 25,651,200 ->25,257,984B reported operation payload. Rank3 cycle1 includes three asynchronous extra events from adjacent work; excluded from exact signature claim. No wire-byte or exposed-time inference.

- `2026-09-26T02:12:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run263`（design-check）。

- `2026-09-26T02:12:09Z` Run `run263` 记录为 `pass`；正确性为 `pass`。Static frozen-path source audit: DSpark-only constructor gate; original MTP method unchanged; borrowed runner binds MTP hidden only when method=mtp; direct Target/DSpark handoffs never read buffer; Run260 FULL Graph rank/cohort gates passed. Not an exact same-state tensor differential.

- `2026-09-26T02:27:40Z` Run `run259` 记录为 `pass`；正确性为 `pass`。Unprofiled frozen48×32K→1024 c12 DSpark7 candidate formal: 48/48 success each; TPS583.892,603.802,594.133, median594.133. 128/128 rank/cohort records pass FULL Graph; borrowed source restored SHA11dd3e... and 8 NPUs idle. Historical Run99 median571.681 is not a contemporaneous control; Run264 will measure same-host unpatched path before KEEP decision.

- `2026-09-26T02:27:40Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run264`（benchmark）。

- `2026-09-26T02:32:31Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run265`（review）。

- `2026-09-26T02:32:31Z` Run `run265` 记录为 `pass`；正确性为 `not-applicable`。Astra High independent read-only review separates compulsory resource from scheduling DAG, rejects current order as semantic bound, identifies private next-metadata scratch overlapping DSpark with parking invalidation as next falsifiable experiment. Executable V1 DAG schema validates both edge sets and leaves numeric bounds null until capacity/node/alias evidence exists.

- `2026-09-26T02:38:40Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run266`（profile）。

- `2026-09-26T02:38:40Z` Run `run266` 记录为 `pass`；正确性为 `not-applicable`。Observed 16 latest full-graph rank-cycle windows: baseline device span54.161ms, compute/copy union39.903ms, HCCL union11.200ms, overlap1.078ms, device gap4.649ms; candidate53.745/38.905/9.376/1.113/4.794ms. Communication AivKernel duplicate rows excluded from compute; reported unions are profiler-perturbed execution activity, not removable critical-path time or hardware/scheduling ceiling.

- `2026-09-26T02:44:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run267`（profile）。

- `2026-09-26T02:44:54Z` Run `run267` 记录为 `pass`；正确性为 `not-applicable`。16/16 current and candidate rank-cycle windows valid; first 98,304B reduce_scatter completion aligns across ranks despite 6-23ms start skew, so first hcom duration includes rank arrival wait and Run266 HCCL union is not a transport floor

- `2026-09-26T02:52:04Z` Run `run264` 记录为 `pass`；正确性为 `pass`。Unpatched same-host control: warmup48 and three formal48 passes 48/48, 128/128 rank-cohort rows pass, FULL Graph; TPS 582.852/574.855/588.301 median582.852; base SHA restored and eight NPUs idle

- `2026-09-26T02:52:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run268`（design-check）。

- `2026-09-26T02:52:05Z` Run `run268` 记录为 `pass`；正确性为 `not-applicable`。Candidate/control medians differ +11.282 tok/s, but per-pass latest-rank runtime normalized to observed cycles is -0.660/-0.093/+0.671 ms per cycle and client residual shifts; exposed wall benefit not established

- `2026-09-26T02:52:34Z` 已记录对比（`comparable=yes`）：Same frozen formal protocol and host; candidate median +11.282 tok/s but sequential runs have trajectory and prefill residual variation, and normalized runtime cost has mixed signs; resource removal causal, E2E benefit unresolved

- `2026-09-26T02:53:35Z` 主控结论为 `PIVOTED`。Causal DSpark-unused MTP stash traffic/collective removal passed correctness, but same-host repeated formal median advantage is confounded; normalized runtime per cycle has mixed signs. Numeric resource and scheduling bounds remain unknown. Pivot to complete execution dependency scheduling rather than promoting an unproved TPS gain.
