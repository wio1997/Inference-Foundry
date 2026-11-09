#!/bin/bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${EVAL_DIR}/.." && pwd)"
cd "${REPO}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
EP="${EP:-8}"
REASONING_EFFORT="${REASONING_EFFORT:-100}"
TAG="${TAG:-${MODEL//\//_}-j${REASONING_EFFORT}}"
DATA_FILE="${DATA_FILE:-${EVAL_DIR}/data/hle_test.jsonl}"
OUT="${EVAL_DIR}/outputs/hle/${TAG}"

echo "[hle] model=${MODEL} ep=${EP} effort=${REASONING_EFFORT} -> ${OUT}"

# 1) inference (single sample per problem, temperature from BENCH_DEFAULTS)
python eval/run_eval.py --bench hle --model-name "${MODEL}" --ep "${EP}" \
    --n-sample 1 --reasoning-effort "${REASONING_EFFORT}" \
    --tag "${TAG}" --data-file "${DATA_FILE}"

# 2) evaluate (LLM judge only)
# need EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY / EVAL_JUDGE_MODEL
python eval/score.py --out-dir "${OUT}" --judge llm --judge-concurrency 16

echo "done: ${OUT}/summary.json"
