#!/usr/bin/env bash
# Verified offline SMG1.13 frontend for the existing native Ascend 4P2D service.
set -eu
ROOT=${REMOTE_ROOT:-/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015}
export PYTHONPATH="$ROOT/optimizations/router-smg113-v1/runtime${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SMG_INSTANCE_TAG=router-smg113-production-v1
exec python3 -B -m smg.launch_router  --pd-disaggregation --policy round_robin  --prefill http://172.16.12.34:31400 31934  --decode http://172.16.12.36:31401  --host 172.16.12.36 --port 31480  --prometheus-host 0.0.0.0 --prometheus-port 10001  --request-timeout-secs 7200  --health-check-timeout-secs 10 --health-check-interval-secs 15  --health-failure-threshold 3 --health-success-threshold 1  --disable-tokenizer-autoload --disable-retries  --log-dir "$ROOT/logs/router-smg113" --log-level info
