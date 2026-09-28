# Run595 — Run99 conditional prefix-active work envelope

The old Run99 formal record contains all8 matching Runtime cohort IDs, cycle counts, per-slot staged output counts and aggregate acceptance window means, but not per-cycle token/count history. Run595 restores each window's exact integer accepted count and solves an integer **outer relaxation**: all12 slots start active; an active cycle emits1–8 accepted tokens; each slot has one active prefix followed by parked zero counts; each slot's staged total and each window's aggregate accepted total equal the saved values. The model deliberately does **not** require parking immediately at1024, so it retains legal Host mirror lag trajectories. All128 Runtime source files are SHA-pinned; the first rank0-only output is preserved as `summary_rank0_provenance_only.json`.

| Formal repeat | Cycles | Previous conditional active Target-8 rows | Prefix-active conditional rows |
|---:|---:|---:|---:|
| 1 | 1,217 | 49,672–116,608 | **79,968–116,608** |
| 2 | 1,212 | 49,672–114,672 | **78,360–114,672** |
| 3 | 1,206 | 49,672–115,144 | **80,856–115,144** |

All24 formal endpoint solves report numerical MILP Optimal with zero gap. Astra High independently checked the all8 source joins, complete enumeration of341,056 small trajectories,48 aggregate comparison cases against MILP, and the actual Run566 A0 per-window feasible assignment; its known 98,496 active rows fall inside the diagnostic envelope. The old 49,672 lower estimate was a valid weaker relaxation, not a contradiction. No independent exact-arithmetic solver certificate was produced.

These rows are **current Target-8 class geometry**, not unique fresh semantic evaluations or mathematically compulsory work. Run569's per-row BF16 `wo_a` 2,885,681,152 conventional ops and routed W4A8 MoE 12,985,565,184 GEMM-equivalent ops yield conditional subset ranges, if every counted active row actually requires the declared dense online evaluation:

| Repeat | `wo_a` conventional ops, T | routed MoE GEMM-equivalent ops, T |
|---:|---:|---:|
| 1 | 230.762–336.494 | 1,038.430–1,514.221 |
| 2 | 226.122–330.907 | 1,017.549–1,489.081 |
| 3 | 233.325–332.269 | 1,049.961–1,495.210 |

The two compute families have different dtypes/service classes and cannot be summed into one FLOP capacity. Actual Target/Draft routing, expert reuse, prefill/seed, KV, HCCL and the required subset of rejected verification work remain unresolved. Even this tighter Run99 range cannot be divided by a chip slogan or isolated attained point to make a strict Hardware or Product TPS endpoint. Scheduling requires a fixed-work all8 dependency graph; exact-board cumulative capacity `C⁺/B` and compulsory traffic remain missing. Formal Current remains **571.681 tok/s**, with strict Resource/Hardware, Scheduling/Execution, Product and numerical Current→limit endpoints null.
