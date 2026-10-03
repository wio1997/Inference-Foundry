### SLO result — concurrency 4

- successful requests N = 4   (output_len_ok=True, all_succeeded=True)
- P99 sample note: N=4: P99 is based on fewer than 100 requests and is not a robust tail estimate

| metric | P50 | P75 | P90 | P99 | target | verdict |
|---|---|---|---|---|---|---|
| TTFT (ms) | 8944.2 | 9109.4 | 9400.5 | 9575.2 | P50<4000 P75<8000 P90<12000 P99<30000 | FAIL/FAIL/PASS/PASS |
| TPOT (ms) | 17.7 | - | 20.4 | 20.4 | P50<18 P90<40 | PASS/PASS |

**ALL SLO PASS: False**
