#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT="$ROOT/evidence/20260920_qli_pair_baseline"
mkdir -p "$OUT/perf" "$OUT/cold"
CANDIDATE_OUT="$OUT/perf" EXPECTED_FLASHCOMM1=true EXPECTED_DSA_CP=true ALLOW_NONDETERMINISTIC_GOLDEN=1 bash "$ROOT/scripts/run_flashcomm_candidate.sh" > "$OUT/perf/runner.log" 2>&1
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8.jsonl
curl -sS --fail http://127.0.0.1:8080/metrics > "$OUT/cold/metrics_before.txt"
for offset in 24 28; do
    date -u "+cold_${offset}_start=%Y-%m-%dT%H:%M:%SZ" >> "$OUT/cold/times.txt"
    docker exec dsv4ab python3 "$ROOT/scripts/bench.py" --dataset "$DATASET" --out "$OUT/cold/cold4_offset${offset}.json" --offset "$offset" --limit 4 --concurrency 1 --max-tokens 128 > "$OUT/cold/cold4_offset${offset}.log" 2>&1
    date -u "+cold_${offset}_end=%Y-%m-%dT%H:%M:%SZ" >> "$OUT/cold/times.txt"
done
curl -sS --fail http://127.0.0.1:8080/metrics > "$OUT/cold/metrics_after.txt"
