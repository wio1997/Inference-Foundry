#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/host_common.sh"
check_container
curl --fail --silent --show-error "http://127.0.0.1:$SERVICE_PORT/health"
curl --fail --silent --show-error "http://127.0.0.1:$SERVICE_PORT/v1/models" \
  | tee "$DEPLOY_DIR/artifacts/models.json"
curl --fail --silent --show-error --max-time 180 \
  "http://127.0.0.1:$SERVICE_PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  --data '{"model":"DeepSeek-V4.1-Flash-1017","prompt":"The capital of China is","max_tokens":32,"temperature":0,"stream":false}' \
  > "$DEPLOY_DIR/artifacts/completion.json"
python3 - "$DEPLOY_DIR/artifacts/completion.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert 'error' not in d, d
assert d['usage']['completion_tokens'] > 0, d
assert d['choices'][0].get('text'), d
print(json.dumps(d, ensure_ascii=False, indent=2))
PY
curl --fail --silent --show-error "http://127.0.0.1:$SERVICE_PORT/metrics" \
  > "$DEPLOY_DIR/artifacts/metrics.prom"
docker inspect "$CONTAINER_NAME" > "$DEPLOY_DIR/artifacts/container-inspect.json"
npu-smi info > "$DEPLOY_DIR/artifacts/npu-smi.txt"
echo 'HTTP checks passed. Complete the eight-rank and feature checks in the manual.'
