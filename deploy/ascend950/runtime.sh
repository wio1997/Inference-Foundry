#!/usr/bin/env bash
# Packaging helper for the validated functional layout; not a serving benchmark.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
IMAGE=${DEEPINFER_IMAGE:-local/deepinfer:950dt-rebuild-20261021}
ENGINE=deepinfer-950-engine
API=deepinfer-950-api
MODE=${1:---help}
if [ "$MODE" = --help ]; then
  printf '%s\n' 'Usage: runtime.sh --check | --topology | --operator-checks | --start | --start-api'
  printf '%s\n' 'Set MODEL_DIR and HCCL_SOCKET_IFNAME for model startup. Creates only task-owned containers.'
  exit 0
fi
if [ "$MODE" = --check ]; then
  docker ps --filter label=inference-foundry.scope=950-deploy --format '{{.Names}} {{.Image}} {{.Status}}'
  curl --fail-with-body -sS --max-time 10 http://127.0.0.1:18110/readyz
  exit 0
fi
case "$MODE" in --topology|--operator-checks|--start|--start-api) ;; *) echo 'Unknown mode'; exit 2;; esac
docker image inspect "$IMAGE" >/dev/null
test -f "$ROOT/build-state.txt"
grep -q BUILD_COMPLETE_DEVICE_TEST_PENDING "$ROOT/build-state.txt"
test -f "$ROOT/image-lock.json"
python3 - "$ROOT/image-lock.json" "$IMAGE" <<'PY'
import json, subprocess, sys
expected = json.load(open(sys.argv[1]))['image_id']
actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', sys.argv[2]]))[0]['Id']
assert actual == expected, 'Image changed; inspect the build and update its lock deliberately.'
PY

owned_container() {
  python3 - "$1" "$ROOT" "$IMAGE" <<'PY'
import json, subprocess, sys
p = subprocess.run(['docker', 'inspect', sys.argv[1]], capture_output=True)
if p.returncode:
    raise SystemExit(1)
d = json.loads(p.stdout)[0]
assert (d['Config'].get('Labels') or {}).get('inference-foundry.scope') == '950-deploy', 'Container belongs to another task'
assert any(x['Source'] == sys.argv[2] and x['Destination'] == '/work' for x in d['Mounts']), 'Workspace mismatch'
expected = json.loads(subprocess.check_output(['docker','image','inspect',sys.argv[3]]))[0]['Id']
assert d['Image'] == expected, 'Container image mismatch'
print(d['State']['Running'])
PY
}
if [ "$MODE" = --start-api ]; then
  : "${MODEL_DIR:?Set MODEL_DIR to the verified checkpoint directory}"
  if docker container inspect "$API" >/dev/null 2>&1; then
    running=$(owned_container "$API")
    if [ "$running" != True ]; then docker start "$API"; fi
    exit 0
  fi
  docker run -d --name "$API" --label inference-foundry.scope=950-deploy \
    --network host --cpus 2 --memory 4g \
    -v "$ROOT:/work" -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash-1017:ro" \
    -e DEEPINFER_HF_MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017 \
    --entrypoint /work/recipe-venv/bin/python "$IMAGE" \
    -m uvicorn api:app --app-dir /work --host 127.0.0.1 --port 18110
  exit 0
fi

devices=()
for i in $(seq 0 7); do test -e "/dev/davinci$i"; devices+=(--device "/dev/davinci$i"); done
for path in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc /dev/ummu /dev/uburma/* /dev/ubctl; do
  if [ -e "$path" ]; then devices+=(--device "$path"); fi
done
NPU_SMI=$(command -v npu-smi || true)
if [ -z "$NPU_SMI" ]; then NPU_SMI=/usr/local/sbin/npu-smi; fi
test -f "$NPU_SMI"

if [ "$MODE" = --topology ]; then
  docker run --rm --name deepinfer-950-topology --label inference-foundry.scope=950-deploy \
    --network host "${devices[@]}" -v "$ROOT:/work" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
    --entrypoint /bin/bash "$IMAGE" -c \
    'source /usr/local/Ascend/cann-9.2.0/set_env.sh; export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; exec /work/engine-venv/bin/python /work/generate_topology.py'
  exit 0
fi
if [ "$MODE" = --start ] && docker container inspect "$ENGINE" >/dev/null 2>&1; then
  running=$(owned_container "$ENGINE")
  if [ "$running" = True ]; then echo 'Owned engine already running'; exit 0; fi
fi

# Refuse to claim occupied devices. Never stop another service automatically.
mkdir -p "$ROOT/runtime-20261021"
docker run --rm --pid host "${devices[@]}" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$NPU_SMI:/usr/local/bin/npu-smi:ro" \
  --entrypoint /bin/bash "$IMAGE" -c \
  'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; npu-smi info' \
  > "$ROOT/runtime-20261021/npu-preflight.txt"
python3 - "$ROOT/runtime-20261021/npu-preflight.txt" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
assert {int(x) for x in re.findall(r'No running processes found in NPU\s+(\d+)\b', text)} == set(range(8)), \
    'Could not prove all eight NPUs idle. See npu-preflight.txt; no inference was started.'
PY

: "${HCCL_SOCKET_IFNAME:?Set the real host network interface; see ip -br addr}"
test -d "/sys/class/net/$HCCL_SOCKET_IFNAME" || { echo 'HCCL_SOCKET_IFNAME is not a host network interface'; exit 1; }
test -f "$ROOT/runtime-20261021/topology/hccl_rootinfo-local-only.json"
test -d "$ROOT/runtime-20261021/topology/hixlep"
test -f /lib/route.conf
common=(--network host --ipc host --cpus 16 --memory 1t --pids-limit 4096 --ulimit memlock=-1:-1
  --label inference-foundry.scope=950-deploy "${devices[@]}"
  -v "$ROOT:/work" -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v "$NPU_SMI:/usr/local/bin/npu-smi:ro" -v /lib/route.conf:/lib/route.conf:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v "$ROOT/runtime-20261021/topology/hccl_rootinfo-local-only.json:/etc/hccl_rootinfo.json:ro"
  -v "$ROOT/runtime-20261021/topology/hixlep:/etc/hixlep:ro"
  -e DEEPINFER_ACCELERATOR=ascend -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  -e DEEPINFER_UBMEM_NET_LAYER=0 -e DEEPINFER_ENGRAM_UBMEM_ONLY=1
  -e DEEPINFER_USE_MULTIMODAL_POOL=0 -e DEEPINFER_USE_NEXTN_DECODE=0
  -e DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0 -e DEEPINFER_ENFORCE_EAGER=1
  -e DEEPINFER_DFLASH_ENFORCE_EAGER=1 -e EP_AVOID_RECORD_STREAM=1 -e TASK_QUEUE_ENABLE=0
  -e HCCL_SOCKET_IFNAME="$HCCL_SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME=lo -e HCCL_IF_BASE_PORT=49361
  -e OMP_NUM_THREADS=2 -e PYTHONUNBUFFERED=1
  -e EP_JIT_CACHE_DIR=/work/runtime-20261021/ep-jit
  -e DG_JIT_CACHE_DIR=/work/runtime-20261021/dg-jit
  -e TILELANG_CACHE_DIR=/work/runtime-20261021/tilelang-cache)
setup='source /usr/local/Ascend/cann-9.2.0/set_env.sh; export LD_LIBRARY_PATH=/work/engine-venv/lib/python3.12/site-packages/torch/lib:/work/engine-venv/lib/python3.12/site-packages/torch_npu/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; '
if [ "$MODE" = --operator-checks ]; then
  docker run --rm --name deepinfer-950-checks "${common[@]}" --entrypoint /bin/bash "$IMAGE" -c \
    "$setup set -e; /work/engine-venv/bin/python /work/checks/check-flashmla.py; /work/engine-venv/bin/python /work/checks/check-dg-layout.py; /work/engine-venv/bin/python /work/checks/check-candidate-short-context.py; /work/engine-venv/bin/python /work/checks/check-engram-direct.py --use-fp8; /work/engine-venv/bin/python /work/checks/check-engram-direct.py"
  exit 0
fi
: "${MODEL_DIR:?Set MODEL_DIR to the verified checkpoint directory}"
python3 - "$ROOT/model-verification.json" "$MODEL_DIR" "$ROOT/locks/model-manifest.json" <<'PY'
import hashlib, json, pathlib, sys
d = json.load(open(sys.argv[1]))
assert d['phase'] == 'complete' and d['sha256_verified_files'] == 84 and d['weight_shards'] == 48
assert pathlib.Path(d['path']).resolve() == pathlib.Path(sys.argv[2]).resolve(), 'Verified model path differs'
assert d['manifest_sha256'] == hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest(), 'Model manifest changed'
PY
if docker container inspect "$ENGINE" >/dev/null 2>&1; then
  docker start "$ENGINE"
else
  docker run -d --name "$ENGINE" "${common[@]}" \
    -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash-1017:ro" \
    -e DEEPINFER_HF_MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017 \
    --entrypoint /bin/bash "$IMAGE" -c "${setup}exec /work/engine-venv/bin/python /work/engine_bridge.py"
fi
