# C32 L13-D native decode two-step profile: accepted diagnostic result and limits

Status: L13-D `CLOSED_ACCEPTED` for bounded diagnostic evidence only. This note is not a performance certificate and does not supersede the accepted L12D measurement.

- Diagnostic run: `20261019T-L13-sglang-C32-D-native-decode-two-step-profile-v1`
- Capture: `20261019T-L13-sglang-C32-D-two-step-native-v1`
- Evidence: `results/C32-L13-D-two-step-evidence-v1.json` SHA `5fccbbafa1072417625192d6f7f904a6815685f507866787e09526ae9bd768bf`
- Decision: `results/C32-L13-D-diagnostic-acceptance-decision-v1.json` SHA `b4f297c2b8bfa5ecdd0691c0fa3a2100c5091aa683627c64389093753514ce6a`
- Independent auditor: `scripts/audit_C32_L13_D_diagnostic_local_v2.py` SHA `e9754f931295600e24cee438cb811dd04dae82b3e7678691156233f1b8c414ed`
- Config: `config/C32-L13-D-native-decode-v1/D-two-step-diagnostic-v1.json` SHA `035d37267b9895790d0a733a4a219d15d15262382a211064305303b4aa2b9ee2`
- Profile-source proof: profiler manager SHA `a962ab8283b468d6c3f10e53527613ea79569490cfdef340f3c22ffbee6bdcc3`; model-runner step-span source SHA `df648164f669f9bbf823fa63e4ba7d7f3e27205eed301235364809e90baf8ce2`.

## 1. What was actually executed

One C32 diagnostic cohort only: 32 logical requests / 64 backend HTTP attempts, 80,000 input tokens and 600 D output tokens per request. Actual successful output was 19,200 D tokens plus 32 P one-token responses, total 19,232. There were no repeats, no new warm cohort, no model request outside this cohort and no new optimization.

All 32 D workers produced profile data: 16 workers on node 36 and 16 on node 41. Each rank shows the required 2 VERIFY and 10 DRAFT step spans plus 2 outer draft and 2 draft_extend spans. Native collection duration per rank was 0.72441799 s to 0.87582944 s (median 0.8448297600000001 s).

The native metric archive independently re-derived 91 sample pairs / 91 complete pairs, 182 native HTTP rows, 0 errors and 32 productive decode Graph deltas (178 to 296 per global EP rank).

## 2. Raw qualification is preserved, not normalized away

The raw profile control did not pass:

- Start: one attempt, raw HTTP 200.
- Stop: one attempt, raw HTTP 500 `Internal Server Error`.
- Stop error: `AssertionError('Native stop did not acknowledge completion or already-inactive state')`.
- `profile_control.pass` remained false and raw HTTP 500 remains preserved.
- No generic HTTP-500 waiver, retry, replay or repair was applied.

Completion is qualified only by the exact pinned-source duplicate-inactive stop error plus actual all-32 native end-info/current-RID done markers, continued request completion and final idle. The two D nodes each show 16 starts, 16 dones and 32 current-RID markers with 16 worker folders.

Cache evidence is likewise preserved: raw cache observation passes are 31/32; source qualification raises the qualified count to 32. Prefix id 16 has no D storage field, and that value stays null. It is not filled with zero and is not used for a direct D L3-read claim.

PD evidence: 32 P MemCache readers and 32 D SDMA receiving devices. The accepted primary recomputation records 258 PD write batches and 569,509,249,024 descriptor bytes (`send_kvcache` 569,507,840,000 plus `send_aux` 1,409,024). Descriptor bytes are not physical wire bytes; `physical_wire_bytes` remains null. No per-request native transfer join is invented.

## 3. Per-rank statistics derivable from the accepted evidence

The table below reports, for each metric, the minimum, median and maximum of the 32 per-rank values. Every value is computed within one rank from that rank's accepted native rows. These are not sums across ranks and not an end-to-end latency decomposition.

| Per-rank metric | min | median | max | Scope |
|---|---:|---:|---:|---|
| Host VERIFY total, 2 spans (us) | 7,195.81 | 10,262.07 | 16,547.70 | Host operator span total; asynchronous enqueue, not device execution |
| Host DRAFT total, 10 spans (us) | 192,821.63 | 216,025.03 | 233,647.00 | Host operator span total; nested work can overlap |
| Host outer `draft`, 2 spans (us) | 176,381.12 | 200,778.58 | 219,394.06 | Host operator span total |
| Host `draft_extend`, 2 spans (us) | 179,145.08 | 184,149.76 | 188,916.47 | Host operator span total |
| Native API memcpy time (us) | 141,957.74 | 164,810.23 | 170,164.43 | `aclrtMemcpy` / `aclrtMemcpyAsync` sum; waits can drain earlier Graph work |
| Native API memcpy count | 149 | 149 | 213 | Per-rank count |
| Native API sync time (us) | 26,575.60 | 84,896.84 | 110,369.00 | Synchronize-family API sum; not device time |
| Native API sync count | 87 | 87 | 88 | Per-rank count |
| Native API launch time (us) | 44,482.92 | 60,212.91 | 70,464.06 | Launch-family API sum |
| Native API launch count | 2,569 | 2,911 | 2,953 | Per-rank count |
| Kernel recorded-interval union (us) | 257,275.25 | 350,023.88 | 375,921.00 | Union of recorded native kernel intervals |
| Kernel interval coverage | 0.5631 | 0.7708 | 0.8123 | Union divided by recorded window; not physical utilization |
| Task-time recorded-interval union (us) | 260,155.75 | 354,567.63 | 380,246.50 | Union of `task_time.csv` intervals |
| Task-time interval coverage | 0.5695 | 0.7806 | 0.8216 | Union divided by recorded window |
| Kernel COMM-name interval union (us) | 102,715.25 | 125,358.63 | 158,950.50 | Frozen COMM regex only; excludes MC2 MoE dispatch/combine |
| `MoeDistributeDispatchV2` duration sum (us) | 46,646.03 | 119,139.36 | 135,401.83 | `op_statistic.csv` name bucket, 160 rows on every rank |
| `MoeDistributeCombineV2` duration sum (us) | 11,917.44 | 15,164.19 | 22,718.10 | `op_statistic.csv` name bucket, 160 rows on every rank |

These ranges show that host-side DRAFT span time is large and variable per rank, while the host VERIFY span is much smaller. They do not establish which portion is Graph drain, CPU metadata preparation or device execution; the accepted aggregates overlap and cannot separate those causes.

## 4. Mandatory interpretation limits

- **Host VERIFY is not device time.** The host VERIFY span encloses asynchronous enqueue of Graph work. Its recorded host duration cannot be called target-device execution time or used as a TPOT component without separate device-side evidence.
- **Do not add overlapping views.** Host draft/draft_extend spans, native API waits, kernel intervals and task-time intervals overlap. Adding them, or summing across ranks, would double count.
- **COMM regex is not total communication.** The frozen regex covers hcom/hccl/all-to-all/all-gather/all-reduce/reduce-scatter names. It does not include `MoeDistributeDispatchV2` or `MoeDistributeCombineV2`, so the COMM bucket must not be presented as total communication time. The MC2 name buckets are also per-rank `op_statistic` sums, not a network-wire measurement.
- **Kernel interval coverage is partial.** Coverage is about 0.56 to 0.81 of the recorded window. Missing, waiting or unrecorded tasks prevent equating the union with physical NPU utilization or pure computation.
- **Task queue remains pinned guidance, not an observed vLLM value.** SGLang currently runs with `TASK_QUEUE_ENABLE=0`, consistent with the pinned official decode-Graph guidance. The operative vLLM task-queue value is still unverified; do not claim vLLM used 2 and do not change the flag from this diagnostic.
- **Selection is not attribution.** Any row sampling from these CSVs is a bounded context sample. It cannot support a full-phase attribution, a physical-utilization percentage or a causal claim by itself.

## 5. Next step

L14 is planning-only: read the existing L13 trace CSV context and the exact installed speculative preparation sources with zero model requests, zero new profiles and no service reload. Only after that evidence is reviewed should a single optimization candidate be chosen. The draft collector is `scripts/glm53_C32_L14_D_trace_context_v1.py` with wrapper `scripts/inspect_C32_L14_D_trace_context_v1.py`; it is `DRAFT_PENDING_PRIMARY_REVIEW` and has not been executed.
