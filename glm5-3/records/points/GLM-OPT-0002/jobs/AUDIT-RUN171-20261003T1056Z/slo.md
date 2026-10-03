### SLO result — concurrency 2

- successful requests N = 4   (output_len_ok=True, all_succeeded=True)
- P99 sample note: N=4: P99 is based on fewer than 100 requests and is not a robust tail estimate

| metric | P50 | P75 | P90 | P99 | target | verdict |
|---|---|---|---|---|---|---|
| TTFT (ms) | 4466.4 | 5623.3 | 5642.4 | 5653.9 | P50<4000 P75<8000 P90<12000 P99<30000 | FAIL/PASS/PASS/PASS |
| TPOT (ms) | 9.7 | - | 15.0 | 17.0 | P50<18 P90<40 | PASS/PASS |

**ALL SLO PASS: False**
