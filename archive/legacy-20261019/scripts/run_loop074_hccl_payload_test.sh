#!/usr/bin/env bash
set -eo pipefail
ROOT=/data/wio/Inference_Foundry
OUT=${RUN339_OUT:-$ROOT/evidence/20260926_loop074_refill/run339}
TOOL=/usr/local/Ascend/cann-9.1.0/tools/hccl_test
mkdir -p "$OUT"
source /usr/local/Ascend/cann/set_env.sh
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_CONNECT_TIMEOUT=17

# This script runs *inside* the dedicated 8-NPU container, only after service stop.
max_hbm=$(npu-smi info | grep -oE '[0-9]+ */ *65536' | awk -F/ '{gsub(/ /,"",$1); print $1}' | sort -rn | head -1)
if test "${max_hbm:-65536}" -gt 10240; then
    echo "NPUs busy, max HBM ${max_hbm} MB" >&2
    exit 75
fi
cd "$TOOL"
: >"$OUT/cases.tsv"
run_case() {
    local name=$1 binary=$2 bytes=$3 dtype=$4
    set +e
    mpirun --allow-run-as-root --bind-to none -n 8 "./bin/$binary" \
        -p 8 -b "$bytes" -e "$bytes" -d "$dtype" -w 5 -n 30 -c 1 -t 1 \
        >"$OUT/$name.log" 2>&1
    local status=$?
    set -e
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$binary" "$bytes" "$dtype" "$status" >>"$OUT/cases.tsv"
    if test "$status" -ne 0; then
        echo "$name failed, exit $status" >&2
        return "$status"
    fi
}
run_case ag_hidden_bf16 all_gather_test 720896 bfp16
run_case ag_router_fp32 all_gather_test 90112 fp32
run_case ag_extra_bf16 all_gather_test 2883584 bfp16
run_case rs_bf16 reduce_scatter_test 720896 bfp16
run_case a2a_bf16 alltoall_test 720896 bfp16
