#!/usr/bin/env bash
source /workspace/deploy/cann_env.sh
PROVIDER=/workspace/deploy/opp/vendors/customize
test -f "$PROVIDER/op_api/lib/libcust_opapi.so"
export ASCEND_CUSTOM_OPP_PATH="$PROVIDER:/usr/local/Ascend/cann-9.2.0/opp/vendors/customize:/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer"
export LD_LIBRARY_PATH="$PROVIDER/op_api/lib:/usr/local/Ascend/cann-9.2.0/opp/vendors/customize/op_api/lib:/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
export VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL=1
export LOCAL_WORLD_SIZE=8 TASK_QUEUE_ENABLE=1
