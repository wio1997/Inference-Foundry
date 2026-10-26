# Loop 报告：loop-023

- 结论：`REJECTED`
- 决定时间：`2026-09-21T06:53:47Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`invalid`
- 因果结论：`not-identifiable`

## 决定依据

k5 is not a runnable TP8 configuration in the current vLLM 0.26 path: graph shapes must be divisible by both 6 and 8. The service failed during KV/backend initialization, so there is no correctness or performance comparison. With model dspark_block_size>=5, the next smaller valid k below7 does not exist under this invariant.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_loop023_k5_screen/startup_failure.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop023_k5_screen/runner.log`

## 下一步

Restore k7 and test the explicit scheduler warning by raising max_num_batched_tokens from8192 to8288, which restores max_num_scheduled_tokens from8096 to8192 while retaining 96 draft slots.
