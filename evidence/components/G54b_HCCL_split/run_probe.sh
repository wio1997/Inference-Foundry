#!/usr/bin/env bash
set -euo pipefail
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
export OMP_NUM_THREADS=1
export LL_GRAPH_WIRE=int8
export DEEPEP_HCCL_BUFFSIZE=1152
for variant in full split; do
 if [ "$variant" = full ]; then export HCCL_BUFFSIZE=1152; else export HCCL_BUFFSIZE=200; fi
 export LL_GRAPH_OUT="/task/comm_split_g54b/evidence/$variant"
 mkdir -p "$LL_GRAPH_OUT"
 timeout --signal=TERM --kill-after=30 500 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=<HOST> --master_port=31954 probe/probe_ll_ep8.py
 done
python3 probe/summarize.py
