#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT="$ROOT/evidence/20260920_qli_cpu_max"
mkdir -p "$OUT/verify" "$OUT/perf"
python3 "$ROOT/scripts/run_qli_verify.py" --out "$OUT/verify" --flag "$OUT/verify.flag" > "$OUT/verify/runner.log" 2>&1
LOG="$ROOT/logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_QLICPU-20260920.log"
grep QLI_PARITY_PASS "$LOG" > "$OUT/verify/parity_lines.log"
if grep -q 'QLI CPU/NPU max mismatch' "$LOG"; then echo 'QLI parity mismatch' >&2; exit 1; fi
CANDIDATE_OUT="$OUT/perf" EXPECTED_FLASHCOMM1=true EXPECTED_DSA_CP=true ALLOW_NONDETERMINISTIC_GOLDEN=1 bash "$ROOT/scripts/run_flashcomm_candidate.sh" > "$OUT/perf/runner.log" 2>&1
