#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260927_loop076_bound/run367
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
mkdir -p "${OUT}/counts" "${OUT}/runtime" "${OUT}/profile" "${OUT}/dag"
cleanup() {
    bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"${OUT}/stop.log" 2>&1 || true
    python3 "${ROOT}/scripts/loop039_live_counts_patch_run121.py" restore --record "${OUT}/restore.json" >"${OUT}/restore.log" 2>&1 || true
    sha256sum /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py "${ROOT}/runtime/extreme_decode.py" >"${OUT}/source_after.sha256"
}
trap cleanup EXIT
sha256sum /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py "${ROOT}/runtime/extreme_decode.py" >"${OUT}/source_before.sha256"
python3 "${ROOT}/scripts/loop039_live_counts_patch_run121.py" install --record "${OUT}/install.json"
docker exec \
    -e MAX_MODEL_LEN=1048576 \
    -e RUN_TS=LOOP076-RUN367 \
    -e EXTREME_RUNTIME_RUN_DIR="${OUT}/runtime" \
    -e EXTREME_RUNTIME_SERVE=1 \
    -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
    -e EXTREME_NATIVE_TARGET_METADATA=1 \
    -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
    -e EXTREME_DSPARK_SLOT_REFRESH=1 \
    -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
    -e EXTREME_GMM_LIVE_COUNTS_DIR="${OUT}/counts" \
    -e EXTREME_RUNTIME_PROFILE_SCOPES=1 \
    -e EXTREME_RUNTIME_CYCLE_PROFILE_DIR="${OUT}/profile" \
    -e EXTREME_RUNTIME_CYCLE_PROFILE_START=64 \
    -e EXTREME_RUNTIME_CYCLE_PROFILE_COUNT=2 \
    -e EXTREME_RUNTIME_CYCLE_PROFILE_SYNC_TARGET=1 \
    -e EXTREME_RUNTIME_CYCLE_PROFILE_MEMORY_ACCESS=1 \
    -e EXTREME_RUNTIME_PROFILE_DAG=1 \
    -e EXTREME_RUNTIME_DAG_DIR="${OUT}/dag" \
    "${CONTAINER}" bash -lc '
        set -e
        cd /data/wio/Inference_Foundry
        bash scripts/serve.sh >"'"${OUT}"'/launcher.log" 2>&1
        for i in $(seq 1 180); do
            if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
            sleep 10
        done
        curl -fsS http://127.0.0.1:8080/health >/dev/null
        python3 scripts/bench.py --dataset "'"${DATASET}"'" --out "'"${OUT}"'/warmup48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"'"${OUT}"'/warmup48.log" 2>&1
        python3 scripts/bench.py --dataset "'"${DATASET}"'" --out "'"${OUT}"'/bench.json" --limit 12 --concurrency 12 --max-tokens 1024 >"'"${OUT}"'/bench.log" 2>&1
    '
sha256sum /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py "${ROOT}/runtime/extreme_decode.py" >"${OUT}/source_patched.sha256"
