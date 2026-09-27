#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260927_loop078_bound/run403
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
mkdir -p "${OUT}/capture" "${OUT}/runtime"
RUN_STARTED=0
cleanup() {
    if test "${RUN_STARTED}" -eq 1; then
        bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"${OUT}/stop.log" 2>&1 || true
    fi
    if test -e "${OUT}/patch_state/manifest.json"; then
        python3 "${ROOT}/scripts/loop078_route_capture_patch.py" restore \
            --state-dir "${OUT}/patch_state" --record "${OUT}/restore.json" \
            >"${OUT}/restore.log" 2>&1 || true
    fi
    sha256sum \
        /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/compilation/acl_graph.py \
        /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py \
        /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py \
        /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py \
        "${ROOT}/runtime/extreme_decode.py" "${ROOT}/runtime/fixed_serving.py" \
        >"${OUT}/source_after.sha256"
}
if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "service already healthy" >&2; exit 1
fi
trap cleanup EXIT
python3 "${ROOT}/scripts/loop078_route_capture_patch.py" check --record "${OUT}/source_check.json"
sha256sum \
    /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/compilation/acl_graph.py \
    /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py \
    /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py \
    /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py \
    "${ROOT}/runtime/extreme_decode.py" "${ROOT}/runtime/fixed_serving.py" \
    >"${OUT}/source_before.sha256"
python3 "${ROOT}/scripts/loop078_route_capture_patch.py" install \
    --state-dir "${OUT}/patch_state" --record "${OUT}/install.json"
RUN_STARTED=1
docker exec \
    -e MAX_MODEL_LEN=1048576 \
    -e RUN_TS=LOOP078-RUN403 \
    -e EXTREME_RUNTIME_RUN_DIR="${OUT}/runtime" \
    -e EXTREME_RUNTIME_SERVE=1 \
    -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
    -e EXTREME_NATIVE_TARGET_METADATA=1 \
    -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
    -e EXTREME_DSPARK_SLOT_REFRESH=1 \
    -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
    -e EXTREME_BOUND_TOKEN_CAPTURE_DIR="${OUT}/capture" \
    "${CONTAINER}" bash -lc '
        set -e
        cd /data/wio/Inference_Foundry
        bash scripts/serve.sh >"'${OUT}'/launcher.log" 2>&1
        for i in $(seq 1 180); do
            if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
            sleep 10
        done
        curl -fsS http://127.0.0.1:8080/health >/dev/null
        python3 scripts/bench.py --dataset "'${DATASET}'" --out "'${OUT}'/warmup48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"'${OUT}'/warmup48.log" 2>&1
        python3 scripts/bench.py --dataset "'${DATASET}'" --out "'${OUT}'/bench.json" --limit 12 --concurrency 12 --max-tokens 1024 >"'${OUT}'/bench.log" 2>&1
    '
POSTS=$(grep -c "POST /v1/chat/completions" "${ROOT}/logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP078-RUN403.log")
echo "${POSTS}" >"${OUT}/server_post_count.txt"
test "${POSTS}" -eq 60
