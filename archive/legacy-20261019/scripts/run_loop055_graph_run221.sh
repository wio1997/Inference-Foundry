#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260925_loop055_moe_replay/run221
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
mkdir -p "$OUT/fingerprints" "$OUT/runtime"
cleanup() {
  rm -f "$OUT/tag.txt"
  bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true
  python3 "$ROOT/scripts/loop055_run221_stage_patch.py" restore --record "$OUT/patch_restore.json" >"$OUT/restore.log" 2>&1 || true
}
trap cleanup EXIT
python3 "$ROOT/scripts/loop055_run221_stage_patch.py" install --record "$OUT/patch_install.json" >"$OUT/install.log"
docker exec "$CONTAINER" bash -lc 'python3 -c "import vllm.model_executor.layers.fused_moe.runner.moe_runner; import vllm_ascend.ops.dsa; import vllm_ascend.worker.model_runner_v1; print(1)"' >"$OUT/preflight.log" 2>&1
docker exec \
  -e MAX_MODEL_LEN=1048576 \
  -e RUN_TS=LOOP055-RUN221 \
  -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" \
  -e EXTREME_RUNTIME_SERVE=1 \
  -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
  -e EXTREME_NATIVE_TARGET_METADATA=1 \
  -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
  -e EXTREME_DSPARK_SLOT_REFRESH=1 \
  -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
  -e EXTREME_RUN221_OUT_DIR="$OUT/fingerprints" \
  -e EXTREME_RUN221_TAG_FILE="$OUT/tag.txt" \
  -e OUT="$OUT" -e DATASET="$DATASET" \
  "$CONTAINER" bash -lc '
    set -e
    cd /data/wio/Inference_Foundry
    bash scripts/serve.sh >"$OUT/launcher.log" 2>&1
    for i in $(seq 1 180); do
      if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
      sleep 10
    done
    curl -fsS http://127.0.0.1:8080/health >/dev/null
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/warmup48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"$OUT/warmup48.log" 2>&1
    printf A > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/A.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/A.log" 2>&1
    printf G1 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/G1.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/G1.log" 2>&1
    printf E1 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/E1.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/E1.log" 2>&1
    printf E2 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/E2.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/E2.log" 2>&1
    printf G2 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/G2.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/G2.log" 2>&1
    printf G3 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/G3.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/G3.log" 2>&1
    printf E3 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/E3.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/E3.log" 2>&1
    printf E4 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/E4.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/E4.log" 2>&1
    printf G4 > "$OUT/tag.txt"
    python3 scripts/bench.py --dataset "$DATASET" --out "$OUT/G4.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/G4.log" 2>&1
    rm -f "$OUT/tag.txt"
  '
