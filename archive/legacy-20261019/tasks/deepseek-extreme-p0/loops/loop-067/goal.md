# Loop 目标：Typed cache liveness and scheduling ownership

- Loop ID：`loop-067`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T11:51:31Z`

## 目标/假设

Layer2 whole-producer owner scheduling can be evaluated only after typed Target/Draft storage aliases, physical page epochs and protected reads are distinguished from allocation reuse

## 允许修改的路径

- `diagnostics/storage_liveness_census.py`
- `bootstrap/vllm_extreme_handoff.py`
- `scripts/run299_storage_census.sh`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

All eight original FULL Graph ranks pass c12×1024; complete typed leaf registry and selected adjacent-cycle/parking metadata permit classified interval risks without inventing compulsory traffic

## 否定条件

Missing typed leaves, invalid storage intervals, incomplete rank coverage or provenance too weak to classify any owner risk

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
