#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260924_loop035_diagnostic/run87
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
mkdir -p "${OUT}/runtime"
export MAX_MODEL_LEN=1048576
export RUN_TS=LOOP035-RUN87
export EXTREME_RUNTIME_RUN_DIR=${OUT}/runtime
export EXTREME_RUNTIME_SERVE=1
export EXTREME_RUNTIME_RESERVE_TOKENS=1088
export EXTREME_NATIVE_TARGET_METADATA=1
export EXTREME_RUNTIME_TARGET_GRAPH=1
export EXTREME_DSPARK_SLOT_REFRESH=1
export EXTREME_RUNTIME_PROFILE_DAG=1
export EXTREME_RUNTIME_DAG_DIR=${OUT}/dag
cd "${ROOT}"
bash scripts/serve.sh >"${OUT}/launcher.log" 2>&1
for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
        break
    fi
    sleep 10
done
curl -fsS http://127.0.0.1:8080/health >/dev/null
python3 scripts/bench.py --dataset "${DATASET}" --out "${OUT}/bench.json" --limit 12 --concurrency 12 --max-tokens 1024 >"${OUT}/bench.log" 2>&1
