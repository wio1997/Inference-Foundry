# Run649 — reuse complete same-W₀ Product ledgers

Read-only audit of prior Run602 and Run606. No live service or NPU run. The reducer checks each run's SHA-pinned Basis/Product/client admission, clean exits, all8 measured cohort cycle agreement, full per-cycle token/count/draft/branch history lengths, final output conservation, and client c12/1024 wall. Astra High independently rehashed Run606's Product manifest and recomputed all 48 Scheduler pre+accepted bulk→post→API delta token-ID lists. Its 281 manifest SHA references normalize to 280 unique files because `client_admission.json` occurs under both absolute and relative paths; all referenced bytes matched.

| Diagnostic W₀ | Cycles by measured cohort | Total cycles | Final API/client IDs | Diagnostic Product wall | Diagnostic TPS |
|---|---|---:|---:|---:|---:|
| Run602 | 281/286/299/323 | 1189 | 49,152 | 91.483702 s | 537.276 |
| Run606 | 297/309/310/298 | 1214 | 49,152 | 83.395754 s | 589.383 |

Run606 has 564 actual Scheduler prebulk IDs plus 48,588 accepted bulk IDs. Run602 has 1,017 plus 48,135. Each W₀ is internally complete for its measured output and Runtime trajectory, and the two W₀s **must not be combined**. Both use synchronous diagnostic collection; their wall/TPS are neither formal performance results nor interchangeable with formal Run99's 1206-cycle median sample and 571.681 tok/s.

**Bound effect:** P0 final Scheduler/API IDs, full per-cycle Runtime workload and diagnostic Product wall already exist. A new live run solely for these fields would repeat work. The first parameterized Framework-only instance should use Run606's 1214-cycle ledger and its 83.396 s diagnostic Current as an upper witness within that observer configuration. Its lower makespan remains unquantified until fixed-primitive cost intervals and necessary ready/issue/collective/resource edges are attached. The next bounded acquisition should target only value version/readiness, Host enqueue→native and all8 first-collective completion at representative critical transitions, plus observer/cost transfer; ordinary prefill/seed/KV work and Product publication need explicit edges. No numerical best-legal Product TPS ceiling or Current→Bound gap is claimed.

Artifacts: `product_ledger_reuse.json`, `scripts/loop081_framework_product_reuse_run649.py`, Run602/606 original admissions and summaries. Formal Current remains 571.681 tok/s.
