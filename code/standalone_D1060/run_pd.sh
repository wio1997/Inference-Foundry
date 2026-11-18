#!/bin/bash
set -euo pipefail
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
cd /task/sg_pd_g1060
export PYTHONPATH="/task/sg_pd_g1060${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -u launch_pd.py
