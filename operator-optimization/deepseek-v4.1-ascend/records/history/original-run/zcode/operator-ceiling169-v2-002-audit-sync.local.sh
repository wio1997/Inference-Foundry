#!/bin/bash
set -euo pipefail
trap 'task_rc=$?; echo "$task_rc" > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-002-audit-sync.exit-code; exit "$task_rc"' EXIT
{
python3 /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/sync_operator_ceiling_v2.py
} > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-002-audit-sync.output.txt 2>&1
cat /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-002-audit-sync.output.txt
