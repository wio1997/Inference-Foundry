#!/bin/bash
set -euo pipefail
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
cd /task/sg_pd_g523
export PYTHONPATH="/task/sg_pd_g523${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -u launch_pd.py
