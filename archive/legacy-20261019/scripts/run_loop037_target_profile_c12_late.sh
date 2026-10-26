#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260924_loop037_target/run104
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
mkdir -p "${OUT}"
cd "${ROOT}"
curl -fsS --max-time 3 http://127.0.0.1:8080/health >/dev/null
python3 scripts/bench.py --dataset "${DATASET}" --out "${OUT}/bench.json" --limit 12 --concurrency 12 --max-tokens 1024 >"${OUT}/bench.log" 2>&1 &
bench_pid=$!
sleep 3
date -u +%FT%TZ >"${OUT}/profile_start_request_utc.txt"
curl -fsS -X POST --max-time 120 http://127.0.0.1:8080/start_profile >"${OUT}/profile_start.json"
date -u +%FT%TZ >"${OUT}/profile_active_utc.txt"
sleep 0.5
date -u +%FT%TZ >"${OUT}/profile_stop_request_utc.txt"
curl -fsS -X POST --max-time 1200 http://127.0.0.1:8080/stop_profile >"${OUT}/profile_stop.json"
date -u +%FT%TZ >"${OUT}/profile_stopped_utc.txt"
wait "${bench_pid}"
