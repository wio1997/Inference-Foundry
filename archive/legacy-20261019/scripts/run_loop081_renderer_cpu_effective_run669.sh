#!/usr/bin/env bash
set -euo pipefail
cd /data/wio/Inference_Foundry
test ! -e evidence/20260929_loop081_bound/run669/renderer_cpu_effective.json
timeout 240s docker exec vllm-ascend26-dsv4f-w4a8 python3 /data/wio/Inference_Foundry/scripts/loop081_renderer_cpu_effective_run669.py > evidence/20260929_loop081_bound/run669/renderer_cpu_effective.log 2>&1
