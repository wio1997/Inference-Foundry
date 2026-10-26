# Loop 报告：L010-tokenizer-input-contract

- 结论：`INCONCLUSIVE`
- 决定时间：`2026-10-18T19:07:34Z`
- 模式：`spec`
- 冻结目标是否满足：`yes`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Full unmodified SGLang tokenizer loader/post-load helpers pass in explicit CPU dependency view with optional torch_npu omitted. Eight raw strings,18 special tokens,18 template controls and7 async lifecycle cases verified;2x512/3x32768 immutable candidate inputs generated. Startup errors/unavailable native import retained. HTTP/PD/MTP output, reasoning parser and performance not established; overall remains E0.

## 证据

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L010-R006-tokenizer-cpu/result.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/fixtures/M001-v1/manifest.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L010-R004-tokenizer-cpu/result.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L010-R005-tokenizer-cpu/result.json`

## 下一步

Publish M001 CPU/input-contract evidence; then measure existing dynamic batching parameter contrast with same frozen CPU input groups and A/A controls, keeping overall serving E0.
