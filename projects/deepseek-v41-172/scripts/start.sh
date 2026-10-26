#!/usr/bin/env bash
set -euo pipefail
cd "${WIO_ROOT:?Export WIO_ROOT}/projects/deepseek-v41-172"
source config/serve.env
[[ "$(hostname)" == "$TARGET_HOSTNAME" ]] || { echo 'Wrong target host'; exit 1; }
[[ "$(docker image inspect --format '{{.Id}}' "$VLLM_IMAGE")" == "$VLLM_IMAGE_ID" ]] || { echo 'Image identity changed'; exit 1; }
mkdir -p logs runs cache/{xdg,huggingface,vllm,torchinductor,triton,ascend,ascend-work} home tmp evidence
exec 9>runs/start.lock
flock -n 9 || { echo 'Another launch holds start.lock'; exit 1; }
npu-smi info >runs/npu-before-start.txt
if ! python3 - <<'PY'
import re
s=open('runs/npu-before-start.txt').read()
rows=[x for x in s.splitlines() if re.search(r'\|\s*\d+\s+\d+\s+\d+\s*\|', x)]
if rows:
    print('NPU processes are active; refusing to interrupt them:')
    print('\n'.join(rows))
    raise SystemExit(1)
if s.count('No running processes found in NPU') != 8:
    raise SystemExit('Unable to establish that all eight physical NPUs are idle')
PY
then exit 2; fi
python3 scripts/preflight.py >runs/preflight-latest.txt
if ss -lntH "sport = :$SERVICE_PORT" | grep -q .; then echo 'Service port already occupied'; exit 1; fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then echo 'Named container exists; inspect it before restarting'; exit 1; fi
stamp=$(date +%Y%m%dT%H%M%S%z)
log="$PWD/logs/vllm-$stamp.log"
ln -sfn "$(basename "$log")" logs/current.log
args=(docker run -d --pull=never --runtime=runc --name "$CONTAINER_NAME" --restart=no
  -e WIO_ROOT="$WIO_ROOT" -e TARGET_HOSTNAME="$TARGET_HOSTNAME"
  --network=host --shm-size=256g --log-driver=none
  --cap-add=SYS_NICE --cap-add=IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /etc/hccn.conf:/etc/hccn.conf:ro
  -v "$WIO_ROOT:$WIO_ROOT"
  -v "$MODEL_PATH:$MODEL_PATH:ro"
  -v "$PWD/tmp:/tmp"
  -w "$PWD" --entrypoint bash)
for i in $(seq 0 15); do args+=(--device "/dev/davinci$i"); done
for d in davinci_manager devmm_svm hisi_hdc; do args+=(--device "/dev/$d"); done
args+=("$VLLM_IMAGE" -c "exec bash '$PWD/scripts/serve.sh' >'$log' 2>&1")
printf '%q ' "${args[@]}" >"runs/docker-command-$stamp.sh"
printf '\n' >>"runs/docker-command-$stamp.sh"
"${args[@]}" | tee runs/container-id.txt
printf '%s\n' "$log" >runs/log-path.txt
docker inspect --format '{{json .State}}' "$CONTAINER_NAME"
