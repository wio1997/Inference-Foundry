#!/usr/bin/env bash
set -eo pipefail
ROOT=/data/wio/Inference_Foundry
EVIDENCE="$ROOT/evidence/20260920_baseline"
DATASET=/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl
mkdir -p "$EVIDENCE"
for i in $(seq 1 180); do
  if [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models || true)" = 200 ]; then
    date -u '+ready=%Y-%m-%dT%H:%M:%SZ'
    break
  fi
  if [ "$i" = 180 ]; then echo 'service readiness timeout' >&2; exit 1; fi
  sleep 10
done
for i in 1 2; do
  curl -sS --fail -m 180 -H 'Content-Type: application/json' \
    -d '{"model":"dsv4","messages":[{"role":"user","content":"Return only the number 42."}],"temperature":0,"max_tokens":16}' \
    http://127.0.0.1:8080/v1/chat/completions > "$EVIDENCE/smoke_${i}.json"
done
python3 - "$EVIDENCE" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
texts = []
for i in (1, 2):
    d = json.loads((p / f'smoke_{i}.json').read_text())
    c = d['choices'][0]['message']
    texts.append((c.get('reasoning_content') or '') + (c.get('content') or ''))
assert texts[0] and texts[0] == texts[1], 'smoke output mismatch or empty'
(p / 'smoke_check.json').write_text(json.dumps({'equal': True, 'sha256': hashlib.sha256(texts[0].encode()).hexdigest(), 'text': texts[0]}, indent=2))
print('smoke equal and non-empty')
PY
uptime > "$EVIDENCE/host_load_before.txt"
curl -sS --fail http://127.0.0.1:8080/metrics > "$EVIDENCE/metrics_before.txt"
"$ROOT/scripts/sample_npu.sh" 900 "$EVIDENCE/npu_samples.txt" &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT
for n in 1 2; do
  date -u "+bench_${n}_start=%Y-%m-%dT%H:%M:%SZ"
  docker exec dsv4ab python3 "$ROOT/scripts/bench.py" --dataset "$DATASET" \
    --out "$EVIDENCE/bench48_${n}.json" --limit 48 --concurrency 12 --max-tokens 1024 \
    > "$EVIDENCE/bench48_${n}.log" 2>&1
  cat "$EVIDENCE/bench48_${n}.log"
  date -u "+bench_${n}_end=%Y-%m-%dT%H:%M:%SZ"
done
curl -sS --fail http://127.0.0.1:8080/metrics > "$EVIDENCE/metrics_after.txt"
uptime > "$EVIDENCE/host_load_after.txt"
kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
trap - EXIT
printf 'baseline_complete\n'
