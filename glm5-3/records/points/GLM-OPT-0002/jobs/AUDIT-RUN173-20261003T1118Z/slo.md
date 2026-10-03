### SLO result — concurrency 2

- successful requests N = 4   (output_len_ok=True, all_succeeded=True)
- P99 sample note: N=4: P99 is based on fewer than 100 requests and is not a robust tail estimate

| metric | P50 | P75 | P90 | P99 | target | verdict |
|---|---|---|---|---|---|---|
| TTFT (ms) | 3884.2 | 4164.8 | 4286.0 | 4358.7 | P50<4000 P75<8000 P90<12000 P99<30000 | PASS/PASS/PASS/PASS |
| TPOT (ms) | 9.3 | - | 13.2 | 14.7 | P50<18 P90<40 | PASS/PASS |

**ALL SLO PASS: True**
