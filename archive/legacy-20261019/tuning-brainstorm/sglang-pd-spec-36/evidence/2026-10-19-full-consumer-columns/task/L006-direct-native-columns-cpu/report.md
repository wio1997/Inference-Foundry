# Loop 报告：L006-direct-native-columns-cpu

- 结论：`ACCEPTED`
- 决定时间：`2026-10-18T17:40:12Z`
- 模式：`optimization`
- 冻结目标是否满足：`yes`
- 可比性：`valid`
- 因果结论：`partially-supported`

## 决定依据

Accepted within explicit CPU-fixture scope. Exact native argument lists/order/calls/error codes match; all4 paired effects clear local A/A; reduced tracked-object GC is consistent with removal of tuple/transposition intermediates. Native conversion/copy, transport/NPU/PD are not measured. No production source changed.

## 证据

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L006-R002-candidate/result.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/candidates/L006/direct_native_columns.py`

## 下一步

Publish L005/L006 raw sample indexes and bounded CPU candidate; move to M017/M018 NPU metadata/staging consumer support audit and observed read-only resolved config. Revisit descriptor work after actual PD path/N/R/critical-path profile and native API contract are known.
