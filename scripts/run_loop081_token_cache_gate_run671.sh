#!/usr/bin/env bash
set -euo pipefail
cd /data/wio/Inference_Foundry
test ! -e evidence/20260929_loop081_bound/run671/cpu_gate.json
timeout 180s docker exec vllm-ascend26-dsv4f-w4a8 python3 /data/wio/Inference_Foundry/scripts/loop081_token_cache_gate_run671.py > evidence/20260929_loop081_bound/run671/cpu_gate.log 2>&1
