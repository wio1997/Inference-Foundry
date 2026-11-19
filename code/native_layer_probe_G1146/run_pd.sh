#!/bin/bash
set -euo pipefail
cd /task/vllm_standalone_g1146
export PYTHONPATH="/task/vllm_standalone_g1146${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -u /task/vllm_standalone_g1146/launch_pd.py
