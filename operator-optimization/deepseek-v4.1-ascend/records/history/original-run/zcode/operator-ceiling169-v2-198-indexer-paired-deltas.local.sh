#!/bin/bash
set -euo pipefail
trap 'task_rc=$?; echo "$task_rc" > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-198-indexer-paired-deltas.exit-code; exit "$task_rc"' EXIT
{
test ! -e /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis/indexer-layer-paired-deltas.json && python3 /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis/compare_indexer_layer_deltas.py
} > /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-198-indexer-paired-deltas.output.txt 2>&1
cat /Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-198-indexer-paired-deltas.output.txt
