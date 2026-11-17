#!/bin/bash
set -euo pipefail
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
export OMP_NUM_THREADS=1
exec timeout --signal=TERM --kill-after=30 450 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=32004 probe.py
