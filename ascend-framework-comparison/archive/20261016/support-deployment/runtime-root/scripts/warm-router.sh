#!/usr/bin/env bash
# warm93-memcache-aisbench-20261015 :: native SGLang PD router, node 36, port 31480.
#
# Serves the name glm-5.3-pd on http://172.16.12.36:31480/v1 for the AISBench client. Runs
# INSIDE the owned D container on node 36, after the P leader (34) and D leader (36) are ready.
set -u
ROOT=${REMOTE_ROOT:-/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015}
mkdir -p "$ROOT/logs/router"
exec python3 -m sglang_router.launch_router \
  --pd-disaggregation --policy round_robin \
  --prefill http://172.16.12.34:31400 31934 \
  --decode http://172.16.12.36:31401 \
  --host 172.16.12.36 --port 31480 \
  --request-timeout-secs 7200 \
  --health-check-timeout-secs 10 \
  --health-check-interval-secs 15 \
  --health-failure-threshold 3 \
  --health-success-threshold 1 \
  --log-dir "$ROOT/logs/router" --log-level info
