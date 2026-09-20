#!/usr/bin/env bash
# =============================================================================
# serve.sh —— 容器内启动 vllm serve（无 mooncake 版）
#
# 相对原方案（infermooncake.sh）的唯一改动：
#   1) 删除 MOONCAKE_CONFIG_PATH 环境变量
#   2) 删除 --kv-transfer-config（AscendStoreConnector + backend=mooncake）
#   3) 不再需要先起 mooncake_master
# 其余服务化参数与原方案逐字一致，以便与历史数据可比。
#
# 平台适配（本机 aarch64）：LD_PRELOAD 由 x86_64 路径改为 aarch64 路径。
#
# 用法（在容器内）：bash /data/wio/vllm_ascend_26/scripts/serve.sh
# =============================================================================
# 注：不使用 set -u —— CANN 的 nnal/atb/set_env.sh 会引用未定义的 ZSH_VERSION，
#     在 nounset 下会直接终止 shell（见 docs 中的排查记录）。
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${HERE}/env.sh"

mkdir -p "${LOG_DIR}"

echo "=== 服务启动 ==="
echo "RUN_ID       : ${RUN_ID}"
echo "MODEL_PATH   : ${MODEL_PATH}"
echo "DP x TP      : ${DP} x ${TP} (NPU ${NPU_NUM} 卡)"
echo "HOST_PORT    : ${HOST_PORT}"
echo "SERVE_LOG    : ${SERVE_LOG}"
echo "mooncake     : 已移除（KV 缓存在 HBM，由 --enable-prefix-caching 承担）"

# ---------- 前置检查：NPU 必须空闲（残留显存会导致 gpu-memory-utilization 抢不到内存） ----------
# 注：停止服务后设备释放有延迟，这里先等待一段时间再判定，避免误杀正常启动
MAX_USED=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ */ *65536' | awk -F/ '{gsub(/ /,"",$1); print $1}' | sort -rn | head -1)
if [ -n "${MAX_USED}" ] && [ "${MAX_USED}" -gt 10240 ]; then
    echo "[02] 检测到残留显存（单卡最大 ${MAX_USED}MB），等待释放 ..."
    for _ in $(seq 1 18); do
        sleep 10
        MAX_USED=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ */ *65536' | awk -F/ '{gsub(/ /,"",$1); print $1}' | sort -rn | head -1)
        [ -n "${MAX_USED}" ] && [ "${MAX_USED}" -lt 10240 ] && break
        echo "[02]   仍为 ${MAX_USED}MB ..."
    done
fi
if [ -n "${MAX_USED}" ] && [ "${MAX_USED}" -gt 10240 ]; then
    echo "[02] 错误：NPU 上仍有残留显存（单卡最大 ${MAX_USED}MB），请先执行 99_stop_service.sh 清理" >&2
    exit 1
fi
echo "[02] NPU 空闲检查通过（单卡最大占用 ${MAX_USED:-?}MB）"

# ---------- CANN 环境 ----------
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# ---------- 环境变量（与原方案一致；ASCNED_*/PYTHONHASHSPEED 为原方案拼写，按原样保留） ----------
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_BUFFSIZE=1024
# 注：该变量在 vllm-ascend 0.26.0rc1 中已无源码引用（仅存于 tests/e2e 的 yaml），
#     保留是为与原方案一致；vLLM 会打印 "Unknown vLLM environment variable" 警告，属预期。
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_CONNECT_TIMEOUT=17
export ASCNED_CONNECT_TIMEOUT=10000
export ASCNED_TRANSFER_TIMEOUT=10000
export PYTHONHASHSPEED=0
export ACL_OP_INIT_MODE=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=16384
# 权重加载约 6~7 分钟（169.7GB / 8 卡）+ 首次编译 mega_moe，默认 600s 就绪握手偏紧，放宽到 1800s
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export FLASHCOMM1_ENABLED="${FLASHCOMM1_ENABLED:-true}"
export DSA_CP_ENABLED="${DSA_CP_ENABLED:-true}"
# ASCEND_RT_VISIBLE_DEVICES / SOC_VERSION 由容器 env 传入

# ---------- 启动 ----------
# 参数说明（与 vllm 0.26 的硬约束相关，改动前请先读）：
#   --speculative-config 的 num_speculative_tokens 必须同时满足：
#     1) >= dspark_block_size，本模型 config.json 中 dspark_block_size = 5
#     2) 开启 SP 时 (num_speculative_tokens + 1) 需被 tensor_parallel_size(4) 整除
#   原方案的 3 与折中的 5 均不合法（3 < 5；5+1=6 不能被 4 整除），最小合法值为 7。
# ---------- 诊断用可选参数（默认不生效，冻结配置不受影响） ----------
# PROFILER_DIR 非空时启用 torch_npu profiler，并开启 vLLM 自定义 scope 打点；
# 之后用 POST /start_profile、/stop_profile 控制采集窗口。
if [ -n "${PROFILER_DIR:-}" ]; then
    mkdir -p "${PROFILER_DIR}"
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
    EXTRA_SERVE_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR}\",\"torch_profiler_with_stack\":false}"
    echo "[02] profiler 已启用：dir=${PROFILER_DIR}"
fi

nohup vllm serve "${MODEL_PATH}" \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --quantization ascend \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --max-num-seqs 16 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
    --max-num-batched-tokens 8192 \
    --no-disable-hybrid-kv-cache-manager \
    --max-model-len "${MAX_MODEL_LEN}" \
    --data-parallel-size "${DP}" \
    --tensor-parallel-size "${TP}" \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "enforce_eager": true}' \
    --enable-auto-tool-choice \
    --async-scheduling \
    --enable-prefix-caching \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --host 0.0.0.0 \
    --port "${HOST_PORT}" \
    --block-size 32 \
    --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":true,\"enable_static_kernel\":false},\"enable_flashcomm1\":${FLASHCOMM1_ENABLED},\"enable_dsa_cp\":${DSA_CP_ENABLED},\"enable_cpu_binding\":true,\"multistream_overlap_shared_expert\":true}" \
    ${EXTRA_SERVE_ARGS:-} \
    > "${SERVE_LOG}" 2>&1 &

echo "PID=$!"
echo "启动中，日志：${SERVE_LOG}"
