#!/bin/bash
set -euo pipefail
cd /task/vllm_pd_g630
exec python3 -u /task/vllm_pd_g630/launch_pd.py
