#!/usr/bin/env bash
set -euo pipefail
export WIO_META_ENABLE=v176
export WIO_META_ROOT=/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/joint176
export PYTHONPATH="$WIO_META_ROOT/hook${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$WIO_META_ROOT/serve-original.sh" >>"$WIO_META_ROOT/diagnostic-service.log" 2>&1
