#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/host_common.sh"
check_container
test "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" = true
if ss -H -ltn "sport = :$SERVICE_PORT" | grep -q .; then
  echo "Port $SERVICE_PORT is already in use." >&2; exit 1
fi
# The lock also covers weight loading before the HTTP port is open.
docker exec "$CONTAINER_NAME" bash -c 'command -v flock' >/dev/null
docker exec -d "$CONTAINER_NAME" bash -c \
  'exec flock -n /workspace/deploy/service.lock bash /workspace/deploy/serve.sh >> /workspace/deploy/logs/service.log 2>&1'
echo "Startup submitted: $DEPLOY_DIR/logs/service.log"
