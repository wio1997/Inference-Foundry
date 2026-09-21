# Loop 报告：loop-025

- 结论：`ACCEPTED`
- 决定时间：`2026-09-21T09:07:34Z`
- 模式：`optimization`
- 冻结目标是否满足：`yes`
- 可比性：`valid`
- 因果结论：`supported`

## 决定依据

Matched no-spec control proves k7 DSpark is architecturally valuable:491.698 versus208.047 tok/s (2.363x) and17.554 versus54.333ms TPOT (-67.69%). The difference dwarfs4.3% noise. Retain DSpark; optimize proposer necessary compute/acceptance rather than exit speculative decoding.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_loop025_nospec/comparison.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop025_nospec/run1/decode_c12.json`

## 下一步

Test whether skipping the middle of three DSpark draft layers saves enough proposer time while retaining at least3.15 advanced tokens per cycle; target verification preserves output correctness.
