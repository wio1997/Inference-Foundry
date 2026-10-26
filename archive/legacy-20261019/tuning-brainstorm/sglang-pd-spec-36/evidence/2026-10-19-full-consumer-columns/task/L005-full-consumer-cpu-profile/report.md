# Loop 报告：L005-full-consumer-cpu-profile

- 结论：`ACCEPTED`
- 决定时间：`2026-10-18T17:32:27Z`
- 模式：`baseline`
- 冻结目标是否满足：`yes`
- 可比性：`valid`
- 因果结论：`partially-supported`

## 决定依据

Full Python consumer and API arguments pass oracle. Clean clocks, GC counts and separate phase diagnostics support substantial remaining tuple/transpose CPU work. Focused interwoven AA/AB corroborates fragmented local wall effect; original off-CPU event remains unexplained. No native transfer/NPU/PD baseline.

## 证据

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L005-R001-full-consumer/result.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L005-R002-interleaved-noise/result.json`

## 下一步

L006: compare range+tuple consumer with guarded direct-native-column construction; preserve API list values/order/calls/status and retain all noisy samples.
