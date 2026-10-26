#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260926_loop062_nongmm/run264
mkdir -p "$OUT/runtime"
cleanup() { bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true; }
trap cleanup EXIT
# Exact unpatched source is checked before the matching frozen control.
ACTUAL=$(sha256sum /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py | awk '{print $1}')
[ "$ACTUAL" = 11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247 ]
bash "$ROOT/scripts/run_loop036_static_e2e.sh" "$OUT" >"$OUT/driver.log" 2>&1
