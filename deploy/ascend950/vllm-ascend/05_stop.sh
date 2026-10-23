#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/host_common.sh"
check_container
docker stop --time 120 "$CONTAINER_NAME"
docker inspect --format '{{.Name}} {{.State.Status}} {{.State.FinishedAt}}' "$CONTAINER_NAME"
if ss -H -ltn "sport = :$SERVICE_PORT" | grep -q .; then
  echo "Port $SERVICE_PORT is still in use; identify its owner before proceeding." >&2
  exit 1
fi
