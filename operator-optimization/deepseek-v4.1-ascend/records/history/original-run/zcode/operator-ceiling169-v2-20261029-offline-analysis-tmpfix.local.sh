#!/bin/bash
set -euo pipefail
trap 'task_rc=$?; echo "$task_rc" > /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/zcode/operator-ceiling169-v2-20261029-offline-analysis-tmpfix.exit-code; exit "$task_rc"' EXIT
python3 -u /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/run_offline_analysis.py > /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/zcode/operator-ceiling169-v2-20261029-offline-analysis-tmpfix.output.txt 2>&1
cat /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/zcode/operator-ceiling169-v2-20261029-offline-analysis-tmpfix.output.txt
