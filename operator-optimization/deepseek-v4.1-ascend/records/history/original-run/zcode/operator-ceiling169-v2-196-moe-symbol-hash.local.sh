#!/bin/bash
set -euo pipefail
trap 'task_rc=$?; echo "$task_rc" > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-196-moe-symbol-hash.exit-code; exit "$task_rc"' EXIT
{
test ! -e /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis/fixed-moe-symbol-hashes.json && python3 /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis/check_fixed_moe_symbol_hashes.py > /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis/fixed-moe-symbol-hashes.json
} > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-196-moe-symbol-hash.output.txt 2>&1
cat /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-196-moe-symbol-hash.output.txt
