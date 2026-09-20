# Results

## 2026-09-20 — Loop 002: corrected DP1/TP8 baseline (Loop 001 pivoted after parser failure)

**Outcome: accepted warm-cache baseline, 48/48 on each of three corrected passes.** No performance optimization is claimed yet.

| Pass | Output TPS | Mean TTFT | Mean TPOT | Success |
|---|---:|---:|---:|---:|
| warm 1 | 543.65 tok/s | 1120.88 ms | 19.45 ms | 48/48 |
| warm 2 | 523.15 tok/s | 1372.41 ms | 20.08 ms | 48/48 |
| warm 3 | 545.85 tok/s | 1331.80 ms | 19.73 ms | 48/48 |
| median | 543.65 tok/s | 1331.80 ms | 19.73 ms | — |

Each pass generated 49,152 tokens. Run 1–2 prefix hit rate was 99.75%; DSpark acceptance was 42.01% / 2.94 accepted per draft. Third acceptance was 41.21%. NPU AICore median/p90 were 77%/83% in 5-second snapshots during passes 1–2. The 20.5 tok/s spread between runs 1 and 2 is a practical noise warning; future KEEP needs a clear margin or paired repeats.

Correctness: two deterministic short prompts returned `42` identically. Four 32K prompts with 128 output tokens each were frozen in `evidence/20260920_baseline/golden4.json` for candidate output equality checks. The original first full run had a client parser bug (`reasoning` deltas ignored), so its TTFT/TPOT are INVALID; raw file and corrected 2/2 parser proof are retained. No framework source was modified.

Freeze: container image ID, source commits, model config/index hashes, dataset hash, versions and device preflight in `evidence/20260920_baseline/`. Launch command in `scripts/serve.sh`; measured client in `scripts/bench.py`; runner and analysis in `scripts/run_baseline.sh` and `scripts/analyze_baseline.py`. `logs/` has raw service log on this host. The corrected passes followed a full-dataset warmup from the invalid client pass; for a restarted candidate, run one full pass before official warm-cache measurement.

Next: current-topology profiling and a separate cold-prefill measurement. Rank DSpark, TP8 communication, kernel/HBM and host exposed time from actual traces before implementation.

## 2026-09-20 — Loop 003 diagnostic in progress

Two disjoint uncached 32K→128 c1 groups each passed 4/4; TTFT mean 2662.7 and 2627.2 ms. The second group had 0 cache hits over 131,404 queried tokens. System msprof on device 0 succeeded during a third group (4/4); output is indexed under `evidence/20260920_diagnostic/`, but no per-op kernel timeline was captured. Baseline service was stopped by terminating its own API process; all 8 NPU process lists cleared, container remained running. The same serving configuration has been relaunched with `PROFILING_MODE=dynamic` for current-topology application profiling. No performance KEEP/REJECT yet.

## 2026-09-20 — Loop 003 pivoted after diagnostic diagnostic: TP0 application profile

The service was restarted with frozen DP1/TP8 serving parameters plus `PROFILING_MODE=dynamic`, ready at 16:17:54 UTC. Dynamic `msprof` attached to TP0 worker. First profile attempt used a wrong dataset path and issued no request; its small failure log is retained. The corrected cold profile used four new distinct 32K→128 c1 prompts: 4/4, mean TTFT 2719.5 ms under profiler. A separate exact-repeat warm profile used four 128-token requests at c4: 4/4, 140.94 output tok/s under profiler. Neither is directly comparable to the official 48×1024 c12 baseline.

| TP0 trace | Event span | Compute/copy union | HCCL union | All-op union | Largest HCCL types |
|---|---:|---:|---:|---:|---|
| cold 4×32K→128 c1 | 18.583 s | 10.829 s | 6.386 s | 16.510 s | reduce-scatter 3.588 s, all-gather 1.934 s, all-to-all 0.864 s |
| warm 4×128 c4 | 3.401 s | 1.688 s | 1.165 s | 2.799 s | reduce-scatter 0.577 s, all-gather 0.356 s, all-to-all 0.231 s |

Union times can overlap and do not prove removability. The warm trace has only ~0.054 s compute/HCCL overlap and 0.602 s with no recorded device op. FlashComm1 sequence-parallel reductions occur in layers and an all-gather feeds MTP (`models/deepseek_v4.py:1144` in the bound source). This justifies a single-variable FlashComm1-off experiment, not a performance claim. Raw paths and op-summary hashes are in `evidence/20260920_diagnostic/app_profile_index.json`; summarize again with `scripts/analyze_profile.py`. No performance KEEP has been earned.

## 2026-09-20 — Loop 004 pivoted: invalid FlashComm1-only configuration

The service command confirmed DP1/TP8 and `enable_flashcomm1=false` with all other options frozen. All workers failed before model load with `ValueError: DSA CP requires SP to be enabled`. No correctness or performance request was issued; the waiting benchmark was stopped. This is an invalid intervention, not a REJECT on performance. Exact failure excerpt and full log SHA256 are under `evidence/20260920_flashcomm_off/`. Next loop tests the coupled FlashComm1-off/DSA-CP-off path explicitly.
