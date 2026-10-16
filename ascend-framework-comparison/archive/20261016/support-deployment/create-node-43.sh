#!/usr/bin/env bash
# Fresh-host creation template; not executed during handover.
set -euo pipefail
test "${HANDOVER_CREATE:-0}" = 1 || { echo "Set HANDOVER_CREATE=1 after checking node, mounts and ports" >&2; exit 2; }
if docker inspect sglang-aisbench-43-20261014 >/dev/null 2>&1; then
  test "$(docker inspect --format '{{.Image}}' sglang-aisbench-43-20261014)" = sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7
  echo 'EXISTS: sglang-aisbench-43-20261014; existing state is unchanged; inspect ownership before reuse'
else
  docker run -d --name sglang-aisbench-43-20261014 --network host --ipc private --shm-size 1073741824 -v /data/wio/glm53-w8a8-local:/model:ro -v /data/wio/aisbench-offline-43-20261014:/work -w /work --entrypoint bash sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7 -c 'sleep infinity'
fi
