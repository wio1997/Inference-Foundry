#!/bin/bash
set -euo pipefail
cd /task/vllm_standalone_g1121
exec python3 -u /task/vllm_standalone_g1121/launch_pd.py
