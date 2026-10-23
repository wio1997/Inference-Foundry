#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/host_common.sh"
EXPECTED_IMAGE=sha256:eed33114ff8ab28bb2ae83ca6d56b9fad0725a1e13de784a2922cf2ab28c9274
test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$EXPECTED_IMAGE"
test "$(docker image inspect --format '{{.Architecture}}' "$IMAGE")" = aarch64 || \
  test "$(docker image inspect --format '{{.Architecture}}' "$IMAGE")" = arm64
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo 'Container already exists; choose a new name or inspect the existing deployment.' >&2
  exit 1
fi
for p in "$MODEL_DIR/config.json" "$TOPOLOGY_DIR/hccl_rootinfo.json" \
  "$TOPOLOGY_DIR/hixlep" /usr/local/Ascend/driver /usr/local/sbin/npu-smi \
  /lib/route.conf /etc/ascend_install.info; do
  test -e "$p" || { echo "Missing prerequisite: $p" >&2; exit 1; }
done
ip link show "$HCCL_NIC" >/dev/null
if ss -H -ltn "sport = :$SERVICE_PORT" | grep -q .; then
  echo "Port $SERVICE_PORT is already in use." >&2; exit 1
fi
devices=()
for i in {0..7}; do devices+=("/dev/davinci$i"); done
devices+=(/dev/davinci_manager /dev/hisi_hdc /dev/ummu)
for d in 0 1; do
  for e in 2 3 4 5 6; do devices+=("/dev/uburma/udmac${d}d1e${e}"); done
done
device_args=()
for p in "${devices[@]}"; do
  test -c "$p" || { echo "Missing device: $p" >&2; exit 1; }
  device_args+=(--device "$p:$p:rwm")
done
mkdir -p "$DEPLOY_DIR/logs" "$DEPLOY_DIR/backups" "$DEPLOY_DIR/artifacts"
docker run -d --name "$CONTAINER_NAME" --label "deployment.id=$DEPLOY_LABEL" \
  --network host --ipc host --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  --cpus "$CPU_QUOTA" --cpuset-cpus "$CPUSET" \
  --memory "$MEMORY_LIMIT" --memory-swap "$MEMORY_SWAP_LIMIT" \
  "${device_args[@]}" \
  --mount "type=bind,src=$MODEL_DIR,dst=/models/DeepSeek-V4.1-Flash,readonly" \
  --mount "type=bind,src=$DEPLOY_DIR,dst=/workspace/deploy" \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --mount type=bind,src=/usr/local/sbin/npu-smi,dst=/usr/local/bin/npu-smi,readonly \
  --mount type=bind,src=/lib/route.conf,dst=/lib/route.conf,readonly \
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
  --mount "type=bind,src=$TOPOLOGY_DIR/hccl_rootinfo.json,dst=/etc/hccl_rootinfo.json,readonly" \
  --mount "type=bind,src=$TOPOLOGY_DIR/hixlep,dst=/etc/hixlep,readonly" \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e GLOO_SOCKET_IFNAME=lo -e "HCCL_SOCKET_IFNAME=$HCCL_NIC" \
  -e HCCL_IF_BASE_PORT=49381 -e HCCL_BUFFSIZE=1024 \
  -e LOCAL_WORLD_SIZE=8 -e TASK_QUEUE_ENABLE=1 -e OMP_NUM_THREADS=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True -e PYTHONUNBUFFERED=1 \
  -e "SERVICE_PORT=$SERVICE_PORT" \
  --workdir /workspace/deploy --entrypoint /bin/bash "$IMAGE" -c 'sleep infinity'
docker exec "$CONTAINER_NAME" npu-smi info
