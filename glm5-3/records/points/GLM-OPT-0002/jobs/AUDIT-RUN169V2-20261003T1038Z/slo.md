### SLO result — concurrency 2

- successful requests N = 2   (output_len_ok=True, all_succeeded=True)
- P99 sample note: N=2: P99 is based on fewer than 100 requests and is not a robust tail estimate

| metric | P50 | P75 | P90 | P99 | target | verdict |
|---|---|---|---|---|---|---|
| TTFT (ms) | 4494.0 | 5080.2 | 5431.9 | 5642.9 | P50<4000 P75<8000 P90<12000 P99<30000 | FAIL/PASS/PASS/PASS |
| TPOT (ms) | 41.6 | - | 55.2 | 58.3 | P50<18 P90<40 | FAIL/FAIL |

**ALL SLO PASS: False**
