#!/usr/bin/env bash
set -euo pipefail
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
export OMP_NUM_THREADS=1
exec python3 probe/launch_workers.py
