### SLO result — concurrency 2

- successful requests N = 4   (output_len_ok=True, all_succeeded=True)
- P99 sample note: N=4: P99 is based on fewer than 100 requests and is not a robust tail estimate

| metric | P50 | P75 | P90 | P99 | target | verdict |
|---|---|---|---|---|---|---|
| TTFT (ms) | 5070.8 | 6130.9 | 6360.0 | 6497.4 | P50<4000 P75<8000 P90<12000 P99<30000 | FAIL/PASS/PASS/PASS |
| TPOT (ms) | 22.3 | - | 27.9 | 29.6 | P50<18 P90<40 | FAIL/PASS |

**ALL SLO PASS: False**
