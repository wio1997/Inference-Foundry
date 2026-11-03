#!/bin/bash
set -euo pipefail
trap 'task_rc=$?; echo "$task_rc" > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-247-draft-api-outer.exit-code; exit "$task_rc"' EXIT
{
python3 /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis/run_draft_api_outer_ep8.py
} > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-247-draft-api-outer.output.txt 2>&1
cat /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-247-draft-api-outer.output.txt
