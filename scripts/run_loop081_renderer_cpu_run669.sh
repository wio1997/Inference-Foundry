#!/usr/bin/env bash
set -euo pipefail
cd /data/wio/Inference_Foundry
test ! -e evidence/20260929_loop081_bound/run669/renderer_cpu.json
timeout 240s docker exec vllm-ascend26-dsv4f-w4a8 python3 /data/wio/Inference_Foundry/scripts/loop081_renderer_cpu_run669.py > evidence/20260929_loop081_bound/run669/renderer_cpu.log 2>&1
