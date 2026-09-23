#!/usr/bin/env bash
set -euo pipefail
cd /data/wio/Inference_Foundry
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS http://127.0.0.1:8080/health >/dev/null
docker exec vllm-ascend26-dsv4f-w4a8 bash -lc 'cd /data/wio/Inference_Foundry && python3 scripts/bench.py --dataset /data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl --out evidence/20260923_loop035_diagnostic/run5/bench.json --limit 12 --concurrency 12 --max-tokens 128'
