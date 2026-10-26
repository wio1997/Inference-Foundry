#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=$ROOT/evidence/20260925_loop052_prefill_submission/run200
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
CONTAINER=vllm-ascend26-dsv4f-w4a8
mkdir -p "$OUT/phase" "$OUT/runtime" "$OUT/marks"
cleanup() {
  rm -f "$OUT/arm_phase.txt" "$OUT/arm_candidate.txt"
  bash /data/wio/vllm_ascend_26/scripts/99_stop_service.sh >"$OUT/stop.log" 2>&1 || true
  python3 "$ROOT/scripts/loop052_run200_patch.py" restore --record "$OUT/candidate_restore.json" >"$OUT/candidate_restore.log" 2>&1 || true
  python3 "$ROOT/scripts/loop048_prefill_phase_run186_patch.py" restore --record "$OUT/phase_restore.json" >"$OUT/phase_restore.log" 2>&1 || true
}
trap cleanup EXIT
python3 "$ROOT/scripts/loop048_prefill_phase_run186_patch.py" install --record "$OUT/phase_install.json" >"$OUT/phase_install.log"
python3 "$ROOT/scripts/loop052_run200_patch.py" install --record "$OUT/candidate_install.json" >"$OUT/candidate_install.log"
docker exec "$CONTAINER" bash -lc 'python3 -c "import vllm.model_executor.layers.fused_moe.runner.moe_runner; import vllm_ascend.ops.dsa; import vllm_ascend.ops.fused_moe.fused_moe; print(1)"' >"$OUT/preflight.log" 2>&1
docker exec \
  -e MAX_MODEL_LEN=1048576 \
  -e RUN_TS=LOOP052-RUN200 \
  -e EXTREME_RUNTIME_RUN_DIR="$OUT/runtime" \
  -e EXTREME_RUNTIME_SERVE=1 \
  -e EXTREME_RUNTIME_RESERVE_TOKENS=1088 \
  -e EXTREME_NATIVE_TARGET_METADATA=1 \
  -e EXTREME_RUNTIME_TARGET_GRAPH=1 \
  -e EXTREME_DSPARK_SLOT_REFRESH=1 \
  -e EXTREME_TARGET_METADATA_STATIC_KV_MAX=1 \
  -e EXTREME_RUN186_PHASE_DIR="$OUT/phase" \
  -e EXTREME_RUN186_ARM_FILE="$OUT/arm_phase.txt" \
  -e EXTREME_RUN200_ARM_FILE="$OUT/arm_candidate.txt" \
  -e EXTREME_RUN200_MARK_DIR="$OUT/marks" \
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
    python3 scripts/bench_hash.py --dataset "$DATASET" --out "$OUT/warmup48.json" --limit 48 --concurrency 12 --max-tokens 1024 >"$OUT/warmup48.log" 2>&1
    touch "$OUT/arm_phase.txt"
    python3 scripts/bench_hash.py --dataset "$DATASET" --out "$OUT/A.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/A.log" 2>&1
    mkdir -p "$OUT/phase/A"; cp "$OUT"/phase/rank*.jsonl "$OUT/phase/A/"; rm "$OUT"/phase/rank*.jsonl
    touch "$OUT/arm_candidate.txt"
    python3 scripts/bench_hash.py --dataset "$DATASET" --out "$OUT/B.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/B.log" 2>&1
    rm -f "$OUT/arm_candidate.txt"
    mkdir -p "$OUT/phase/B"; cp "$OUT"/phase/rank*.jsonl "$OUT/phase/B/"; rm "$OUT"/phase/rank*.jsonl
    python3 scripts/bench_hash.py --dataset "$DATASET" --out "$OUT/A2.json" --limit 12 --concurrency 12 --max-tokens 1024 >"$OUT/A2.log" 2>&1
    mkdir -p "$OUT/phase/A2"; cp "$OUT"/phase/rank*.jsonl "$OUT/phase/A2/"; rm "$OUT"/phase/rank*.jsonl
    rm -f "$OUT/arm_phase.txt"
  '
