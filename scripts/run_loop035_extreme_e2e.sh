#!/usr/bin/env bash
set -eo pipefail

ROOT=/data/wio/Inference_Foundry
OUT=${1:-${ROOT}/evidence/20260923_loop035_fixed_serving/e2e}
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl

mkdir -p "${OUT}/runtime"
for _ in $(seq 1 6); do
    snapshot=$(npu-smi info)
    max_used=$(printf '%s\n' "${snapshot}" \
        | grep -oE '[0-9]+ */ *65536' \
        | awk -F/ '{gsub(/ /,"",$1); print $1}' \
        | sort -rn | head -1)
    if printf '%s\n' "${snapshot}" | grep -Eiq 'VLLM|python' \
        || [ "${max_used:-65536}" -gt 10240 ]; then
        echo "NPUs are occupied; refusing to disturb the active workload" >&2
        exit 75
    fi
    sleep 10
done

# The repository and NPU devices are mounted into the frozen vLLM-Ascend 0.26
# container, while the SSH host intentionally has no CANN toolkit. Re-enter
# this exact script in that container after the host-side occupancy gate so the
# benchmark cannot accidentally use a different framework checkout.
if [ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    exec docker exec -i vllm-ascend26-dsv4f-w4a8 \
        bash -lc "cd '${ROOT}' && ./scripts/run_loop035_extreme_e2e.sh '${OUT}'"
fi

export MAX_MODEL_LEN=1048576
export RUN_TS=LOOP035-EXTREME-E2E-$(date +%Y%m%d-%H%M)
export EXTREME_RUNTIME_RUN_DIR=${OUT}/runtime
export EXTREME_RUNTIME_SERVE=1
export EXTREME_RUNTIME_RESERVE_TOKENS=1088
export EXTREME_NATIVE_TARGET_METADATA=1
export EXTREME_RUNTIME_TARGET_GRAPH=1
export EXTREME_DSPARK_SLOT_REFRESH=1

cd "${ROOT}"
bash scripts/serve.sh >"${OUT}/launcher.log" 2>&1

for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8080/health >/dev/null; then
        break
    fi
    sleep 10
done
curl -fsS http://127.0.0.1:8080/health >/dev/null

python3 scripts/bench.py \
    --dataset "${DATASET}" \
    --out "${OUT}/warmup.json" \
    --limit 48 --concurrency 12 --max-tokens 1024 \
    >"${OUT}/warmup.log" 2>&1

for run in 1 2 3; do
    python3 scripts/bench.py \
        --dataset "${DATASET}" \
        --out "${OUT}/bench48_${run}.json" \
        --limit 48 --concurrency 12 --max-tokens 1024 \
        >"${OUT}/bench48_${run}.log" 2>&1
done

python3 scripts/analyze_loop034_e2e.py "${OUT}"
