#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260926_loop060_resource/run246
mkdir -p "$OUT"
cleanup() { bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/host_stop.log" 2>&1 || true; }
trap cleanup EXIT
docker exec vllm-ascend26-dsv4f-w4a8 bash -lc "cd /data/wio/Inference_Foundry && bash scripts/run_loop060_memory_profile.sh"
