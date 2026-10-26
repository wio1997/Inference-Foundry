#!/usr/bin/env bash
# =============================================================================
# vllm_ascend_26 统一环境参数（所有脚本 source 本文件）
# 用途：DeepSeek-V4-Flash-0731-w4a8 在 vllm-ascend v0.26.0rc1 上的服务化与测试
# 命名规范：<stage>_<model>_<quant>_<npu>npu_dp<D>tp<T>_<variant>_<YYYYmmdd-HHMM>
# =============================================================================

# ---------- 镜像与容器 ----------
export IMAGE="quay.io/ascend/vllm-ascend:v0.26.0rc1"
export CONTAINER_NAME="vllm-ascend26-dsv4f-w4a8"

# ---------- 模型 ----------
export MODEL_PATH="/data/yxy/DeepSeek-V4-Flash-0731-w4a8"
export MODEL_TAG="dsv4f-w4a8"
export SERVED_MODEL_NAME="dsv4"

# ---------- 硬件 ----------
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NPU_NUM=8
export SOC_VERSION="ascend910b1"

# ---------- 并行策略 ----------
export DP=1
export TP=8

# ---------- 服务 ----------
export HOST_PORT=8080
# 上下文长度：v1 基线 133072；v2 基线 1048576（用 MAX_MODEL_LEN=1048576 覆盖）
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-133072}"
if [ "${MAX_MODEL_LEN}" -ge 1000000 ]; then
    export MAXLEN_TAG="mlen1M"
else
    export MAXLEN_TAG="mlen${MAX_MODEL_LEN}"
fi

# ---------- 路径 ----------
export PROJECT_ROOT="/data/wio/Inference_Foundry"
export LOG_DIR="${PROJECT_ROOT}/logs"
export RESULT_DIR="${PROJECT_ROOT}/results"
export TOOL_DIR="${PROJECT_ROOT}/tools/aisbench_auto_tools_prefix"
export BENCHMARK_DIR="${PROJECT_ROOT}/tools/benchmark"
export FRAMEWORK_DIR="${PROJECT_ROOT}/framework"
export BASELINE_DIR="${PROJECT_ROOT}/baseline"

# ---------- aisbench 测试参数（默认值与原方案一致，可外部覆盖） ----------
export DATASET_PATH="${PROJECT_ROOT}/datasets"
export AISBENCH_WORK_DIR="${RESULT_DIR}/aisbench_work"
export INPUT_LEN="${INPUT_LEN:-32768}"
export OUTPUT_LEN="${OUTPUT_LEN:-1024}"
export DATA_NUM="${DATA_NUM:-48}"
export CONCURRENCY="${CONCURRENCY:-12}"
export REQUEST_RATE="${REQUEST_RATE:-0}"
export DATASET_TYPE="${DATASET_TYPE:-prefix_cache}"
export REPEAT_RATE="${REPEAT_RATE:-0.9}"
export PREFIX_TEST="${PREFIX_TEST:-1}"
export SEED="${SEED:-1}"

# ---------- 运行标识（供日志/结果命名，可外部覆盖） ----------
export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M)}"
export CASE_TAG="${NPU_NUM}npu_dp${DP}tp${TP}_${MAXLEN_TAG}_nomooncake"
export RUN_ID="${MODEL_TAG}_${CASE_TAG}_${RUN_TS}"

export SERVE_LOG="${LOG_DIR}/serve_${RUN_ID}.log"
export TEST_LOG="${LOG_DIR}/aisbench_prefix_in${INPUT_LEN}_out${OUTPUT_LEN}_n${DATA_NUM}_c${CONCURRENCY}_rr${REPEAT_RATE}_${RUN_ID}.log"
