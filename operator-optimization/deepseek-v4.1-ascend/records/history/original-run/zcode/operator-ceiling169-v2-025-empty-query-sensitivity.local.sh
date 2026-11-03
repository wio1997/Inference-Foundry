#!/bin/bash
set -euo pipefail
trap 'task_rc=$?; echo "$task_rc" > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-025-empty-query-sensitivity.exit-code; exit "$task_rc"' EXIT
{
python3 /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/tools/run_eager_draft_attribution_v3.py
} > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-025-empty-query-sensitivity.output.txt 2>&1
cat /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-025-empty-query-sensitivity.output.txt
