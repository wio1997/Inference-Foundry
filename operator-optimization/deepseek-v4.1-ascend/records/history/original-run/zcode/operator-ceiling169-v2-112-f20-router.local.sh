#!/bin/bash
set -euo pipefail
trap 'task_rc=$?; echo "$task_rc" > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-112-f20-router.exit-code; exit "$task_rc"' EXIT
{
python3 /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/fusion/snapshot_f20_router.py
} > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-112-f20-router.output.txt 2>&1
cat /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-112-f20-router.output.txt
