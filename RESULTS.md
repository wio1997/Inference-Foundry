# Results

## 2026-09-20 — Loop 001: environment freeze and DP1/TP8 baseline

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
