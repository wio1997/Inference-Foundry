### SLO result — concurrency 4

- successful requests N = 4   (output_len_ok=True, all_succeeded=True)
- P99 sample note: N=4: P99 is based on fewer than 100 requests and is not a robust tail estimate

| metric | P50 | P75 | P90 | P99 | target | verdict |
|---|---|---|---|---|---|---|
| TTFT (ms) | 6482.8 | 6725.1 | 6733.0 | 6737.8 | P50<4000 P75<8000 P90<12000 P99<30000 | FAIL/PASS/PASS/PASS |
| TPOT (ms) | 35.3 | - | 39.7 | 40.0 | P50<18 P90<40 | FAIL/PASS |

**ALL SLO PASS: False**
