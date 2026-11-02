#!/bin/bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${EVAL_DIR}/.." && pwd)"
cd "${REPO}"

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
EP="${EP:-8}"
N_SAMPLE="${N_SAMPLE:-8}"
REASONING_EFFORT="${REASONING_EFFORT:-100}"
TAG="${TAG:-${MODEL//\//_}-n${N_SAMPLE}-j${REASONING_EFFORT}}"
DATA_FILE="${DATA_FILE:-${EVAL_DIR}/data/gpqa_diamond.jsonl}"
OUT="${EVAL_DIR}/outputs/gpqa/${TAG}"

echo "[gpqa] model=${MODEL} ep=${EP} n_sample=${N_SAMPLE} effort=${REASONING_EFFORT} -> ${OUT}"

# 1) inference
python eval/run_eval.py --bench gpqa --model-name "${MODEL}" --ep "${EP}" \
    --n-sample "${N_SAMPLE}" --reasoning-effort "${REASONING_EFFORT}" \
    --tag "${TAG}" --data-file "${DATA_FILE}"

# 2) evaluate
# LLM judging: need EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY / EVAL_JUDGE_MODEL
python eval/score.py --out-dir "${OUT}" --judge llm --judge-concurrency 16

# rule judging: use this instead when no LLM judge is available
# python eval/score.py --out-dir "${OUT}" --judge rule

echo "done: ${OUT}/summary.json"
