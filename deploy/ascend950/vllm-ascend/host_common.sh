#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/deploy.env"
test "$(readlink -f "$DEPLOY_DIR")" = "$SCRIPT_DIR" || {
  echo 'Set DEPLOY_DIR to the directory containing these scripts.' >&2; exit 1;
}
DEPLOY_LABEL=vllm-ascend-950dt-ds41
check_container() {
  local owner
  owner=$(docker inspect --format '{{index .Config.Labels "deployment.id"}}' "$CONTAINER_NAME")
  test "$owner" = "$DEPLOY_LABEL" || {
    echo 'Container deployment label does not match.' >&2; return 1;
  }
}
