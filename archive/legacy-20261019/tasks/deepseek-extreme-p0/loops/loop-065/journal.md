# 执行日志

- `2026-09-26T07:33:47Z` Loop 已冻结。下一步：Read actual dsa_cp decode gather/Q source and construct SHA-guarded A0/A1/B one-layer diagnostic

- `2026-09-26T07:35:31Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run288`（test）。

- `2026-09-26T07:49:55Z` Run `run288` 记录为 `invalid`；正确性为 `invalid`。Before handoff or Graph capture, ranks4/6 failed pinned Host allocation in vLLM KV block table (aclrtMallocHostWithCfg 207001); no candidate branch, clients, or scheduling evidence. Service stopped, NPUs idle, borrowed source SHA restored.

- `2026-09-26T07:49:55Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run289`（test）。

- `2026-09-26T08:06:09Z` Run `run289` 记录为 `invalid`；正确性为 `invalid`。Repeated pinned Host allocator failure before handoff/Graph; no H003 branch. Targeted cleanup found 336 orphaned container spawn workers (~230GiB RSS), recovered Host MemAvailable ~190 to ~905GiB. All8 NPUs idle and borrowed SHA restored.

- `2026-09-26T08:06:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run290`（test）。

- `2026-09-26T08:23:46Z` Run `run290` 记录为 `invalid`；正确性为 `invalid`。After orphan cleanup, async-immediate layer2 all8 FULL Graph capture and 60/60 exact1024 clients passed; only 2/5 expected fixed cohorts entered Extreme (16/16 rank-cohort FULL/pass). No same-state numerical or E2E comparison; diagnostic TPS noncomparable. Source restored, NPUs idle.

- `2026-09-26T08:27:27Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run291`（test）。

- `2026-09-26T08:44:10Z` Run `run291` 记录为 `pass`；正确性为 `invalid`。Isolated async-immediate layer2 Graph control: server exactly60 POST, 60/60 exact1024, 40/40 FULL Graph Runtime across five cohorts, all8 candidate capture markers. No A0 same-state numerical gate or formal repeated E2E; diagnostic TPS568.814/589.044 not promoted. Exit0, source restored, NPUs idle.

- `2026-09-26T08:44:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run292`（profile）。

- `2026-09-26T09:05:38Z` Run `run292` 记录为 `pass`；正确性为 `invalid`。B delayed FULL Graph and 60/60 exact outputs; all8 x five cohorts. 16/16 device gather/Q overlap; same-prestate parity and matched endpoint pending

- `2026-09-26T09:06:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run293`（profile）。

- `2026-09-26T09:19:27Z` Run `run293` 记录为 `pass`；正确性为 `invalid`。A1 immediate FULL Graph 60/60 exact; 40/40 rank-cohorts; matched B local WKV and next A2A advance but same-state parity and Product gain pending

- `2026-09-26T09:23:46Z` 主控结论为 `PIVOTED`。H003 delayed hidden gather overlaps local Q on all8 and advances one-layer next AllToAll by paired median10.125us versus async-immediate; full Target/cycle and Product gain unproven, same-state typed parity pending. Run287 owner and parking evidence suggests larger architectural gap
