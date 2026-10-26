#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${ROOT}/evidence/20260924_loop035_diagnostic/run92
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
mkdir -p "${OUT}/runtime" "${OUT}/pages" "${OUT}/slots"
export MAX_MODEL_LEN=1048576
export RUN_TS=LOOP035-RUN92
export EXTREME_RUNTIME_RUN_DIR=${OUT}/runtime
export EXTREME_RUNTIME_RESERVE_TOKENS=1088
export EXTREME_NATIVE_TARGET_METADATA=1
export EXTREME_RUNTIME_TARGET_GRAPH=1
export EXTREME_DSPARK_SLOT_REFRESH=1
export EXTREME_STOCK_TARGET_ABA=1
export EXTREME_STOCK_TARGET_ABA_CYCLE=256
export EXTREME_STOCK_TARGET_ABA_REPEAT=1
export EXTREME_TARGET_PAGE_AUDIT_DIR=${OUT}/pages
export EXTREME_TARGET_PAGE_AUDIT_CYCLES=1
export EXTREME_TARGET_SELF_REPLAY=1
export EXTREME_COMPRESSOR_SLOT_VERIFY=1
export EXTREME_KV_SLOT_AUDIT_DIR=${OUT}/slots
cd "${ROOT}"
bash scripts/serve.sh >"${OUT}/launcher.log" 2>&1
for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
        break
    fi
    sleep 10
done
curl -fsS http://127.0.0.1:8080/health >/dev/null
python3 scripts/bench.py --dataset "${DATASET}" --out "${OUT}/bench.json" --limit 12 --concurrency 12 --max-tokens 2048 >"${OUT}/bench.log" 2>&1 || true
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/data/wio/Inference_Foundry/evidence/20260924_loop035_diagnostic/run92")
files = sorted((root / "runtime").glob("rank*.json"))
if len(files) != 8:
    raise RuntimeError(f"strict Stock ABA needs eight rank records, found {len(files)}")
rows = [json.loads(p.read_text()) for p in files]
if not all(r["sampled_stock_fixed_cycle"] == 256 for r in rows):
    raise RuntimeError("Stock continuous cycle sample did not reach 128")
if not all(r["pass"] for r in rows):
    raise RuntimeError("strict Stock ABA restore failed")
if not all(set(r["strict_coverage"]["captured_names"]) for r in rows):
    raise RuntimeError("strict Stock ABA missing page snapshot")
print("strict Stock ABA rank records:", len(rows))
PY
