# Loop 目标：Cohort completion barrier and FIFO slot-refill scheduling bound

- Loop ID：`loop-074`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T18:44:08Z`

## 目标/假设

The fixed 12-slot cohort publication barrier leaves completed slots idle while 48-request c12 client work remains; conditional per-request release/refill could shorten Product E2E if incremental prefill, seed, state and Graph costs fit the available scheduling window

## 允许修改的路径

- `scripts`
- `evidence/20260926_loop074_refill`
- `performance_knowledge`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Pinned history searched; original Run287 all8 48-request completion trace closed; FIFO and ideal conditional cycle models plus order/refill-cost sensitivity validated; next live semantic gate chosen only if robust opportunity survives

## 否定条件

Conditional zero-cost FIFO has no material cycle opportunity, or conservative incremental costs and state constraints consume it; do not infer Product TPS from slot cycles

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
