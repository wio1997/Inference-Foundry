# 执行日志

- `2026-09-26T14:03:40Z` Loop 已冻结。下一步：Run309 SHA/source and contract preflight; Run310 original FULL Graph content baseline; Run311 reversible owner candidate and same-prompt content/rank gate

- `2026-09-26T14:03:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run309`（design-check）。

- `2026-09-26T14:03:41Z` Run `run309` 记录为 `pass`；正确性为 `not-applicable`。Reversible patch anchors and source SHA pass; content hash and FULL Graph gate scripts compiled; live semantics untested

- `2026-09-26T14:03:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run310`（test）。

- `2026-09-26T14:39:28Z` Run `run310` 记录为 `pass`；正确性为 `pass`。Original path content baseline: 12/12x1024 client, 8/8 FULL Graph Runtime, exit0, source original and cards idle; single-cohort diagnostic, not formal E2E

- `2026-09-26T14:39:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run311`（test）。

- `2026-09-26T14:39:36Z` Run `run311` 记录为 `invalid`；正确性为 `invalid`。12/12x1024 client and 8/8 FULL Graph Runtime passed; live owner capture/handoff/replay keys recorded all8. Cross-service content differs (12 reasoning hashes, one content); cycles306 vs original315, so semantic and performance attribution unresolved. Harness exit1 from unbound ROOT during offline compare; after offline comparison content gate fails. All4 borrowed sources restored SHA, service stopped/cards idle. Run312 original repeat control pending.

- `2026-09-26T14:39:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run312`（benchmark）。

- `2026-09-26T14:52:55Z` Run `run312` 记录为 `pass`；正确性为 `pass`。Original-path repeated content control: 12/12x1024 and 8/8 FULL Graph Runtime exit0, 290 cycles. Against Run310 original (315 cycles), all12 reasoning hashes differ and 2 content hashes differ; cross-service content is nondeterministic under this diagnostic. Candidate Run311 content mismatch cannot be causal-attributed without same-prestate comparison.

- `2026-09-26T14:56:56Z` 主控结论为 `PIVOTED`。Run311 integrates owner16 full producer in all8 FULL Graph Runtime ranks and completes 12x1024, but content differs cross-service; Run310 vs unchanged original Run312 also differs in all12 reasoning hashes and cycles315 vs290. Candidate correctness and wall benefit are not identifiable. Run311 wall/cycle 56.810ms exceeds original 56.545/56.643ms; no Product gain promoted. Next isolate dynamic metadata inside Graph on same prestate.
