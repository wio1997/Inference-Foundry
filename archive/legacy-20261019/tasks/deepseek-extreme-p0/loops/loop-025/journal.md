# 执行日志

- `2026-09-21T08:05:27Z` Loop 已冻结。下一步：Create a launcher removing only speculative-config, run the same correctness/warmup/12x32K-to-512 c12 screen, and compare to k7 screens before any full benchmark.

- `2026-09-21T08:06:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `nospec-screen-20260921`（benchmark）。

- `2026-09-21T09:06:01Z` Run `nospec-screen-20260921` 记录为 `pass`；正确性为 `pass`。Matched target-only control passed functional and12/12 c12. It delivered208.047 tok/s, TTFT1644.40ms, TPOT54.333ms versus k7 Loop020491.698 tok/s and17.554ms TPOT. k7 is2.363x throughput and67.69% lower TPOT, with20.52% worse TTFT in this short screen. DSpark is decisively net beneficial.

- `2026-09-21T09:07:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `k7-reference-loop020`（benchmark）。

- `2026-09-21T09:07:34Z` Run `k7-reference-loop020` 记录为 `pass`；正确性为 `pass`。Saved same-runner k7 control from Loop020:12/12,491.698 tok/s, TPOT17.554ms, TTFT1981.90ms.

- `2026-09-21T09:07:34Z` 已记录对比（`comparable=yes`）：Same runner/workload: k7 DSpark491.698 vs target-only208.047 tok/s (+136.34%); TPOT17.554 vs54.333ms (-67.69%). Both12/12 and correctness pass.

- `2026-09-21T09:07:34Z` 主控结论为 `ACCEPTED`。Matched no-spec control proves k7 DSpark is architecturally valuable:491.698 versus208.047 tok/s (2.363x) and17.554 versus54.333ms TPOT (-67.69%). The difference dwarfs4.3% noise. Retain DSpark; optimize proposer necessary compute/acceptance rather than exit speculative decoding.
