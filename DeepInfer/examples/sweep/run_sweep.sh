#!/bin/bash
# 扫一维 batch 参数，出一条吞吐-延迟 Pareto 曲线；结果与图都在 sweeps/<tag>/。
#
#   run_sweep.sh decode  <tag> <context_length> <nextn> <draft_acceptance> [benchmark.py 参数...]
#   run_sweep.sh prefill <tag> <context_length> [benchmark.py 参数...]
#   run_sweep.sh encoder <tag> [ve_sweep.py 参数...]
#
# 位置参数之后的内容透传给 benchmark.py / ve_sweep.py，如 --model_name、--cp。
set -uo pipefail

MODE="${1:?usage: run_sweep.sh <decode|prefill|encoder> <tag> [mode 位置参数...]}"
TAG="${2:?missing <tag>}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT="${REPO_ROOT}/sweeps/${TAG}"
RESULTS="${OUT}/results.jsonl"
mkdir -p "${OUT}"

# benchmark.py 在模块级读 WORLD_SIZE 推 ep_size，多机由调度器注入，单机自己补 1
export WORLD_SIZE="${WORLD_SIZE:-1}"
# 每个扫描点换一个端口：上一点的 TCPStore 退出后端口仍在 TIME_WAIT，复用会卡住 rendezvous。
# 偏移取扫描点下标，各节点算出的端口因此一致。
BASE_PORT="${MASTER_PORT:-29500}"

DESC="mode=${MODE} tag=${TAG}"
case "${MODE}" in
    decode)
        CONTEXT="${1:?decode 模式需要 <context_length>}"
        NEXTN="${2:?decode 模式需要 <nextn>}"
        ACCEPT="${3:?decode 模式需要 <draft_acceptance>}"
        shift 3
        export DEEPINFER_USE_NEXTN_DECODE="${NEXTN}"
        SWEEP_FLAG="--decode_num_requests"
        SWEEP_VALUES=(8 16 32 64 128 256)
        # 长 context 下 decode 读数只取决于已有 KV 的长度，非缓存后缀给足一个 chunk 即可
        EXTRA_ARGS=(--context_length "${CONTEXT}" --non_cached_tokens 4096)
        DESC="${DESC} context=${CONTEXT} nextn=${NEXTN}"
        ;;
    prefill)
        CONTEXT="${1:?prefill 模式需要 <context_length>}"
        shift 1
        SWEEP_FLAG="--prefill_batch_size"
        SWEEP_VALUES=(4096 8192 16384 32768 65536)
        # YoCo 截断固定取 benchmark.py 的默认 140，不作为扫描维度
        EXTRA_ARGS=(--context_length "${CONTEXT}" --yoco_clip)
        DESC="${DESC} context=${CONTEXT}"
        ;;
    encoder)
        # 扫描点与图片尺寸都是 ve_sweep.py 的默认值，要改直接透传 --num_images / --image_size
        ;;
    *)
        echo "未知 mode=${MODE}（只支持 decode / prefill / encoder）" >&2
        exit 1
        ;;
esac

echo "=== sweep ${DESC} -> ${OUT}"
git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" log --oneline -1
echo "node=$(hostname) WORLD_SIZE=${WORLD_SIZE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

FAILED=0
if [ "${MODE}" = "encoder" ]; then
    echo "--- VE sweep ($(date +%H:%M:%S)) ---"
    python -u "${SCRIPT_DIR}/ve_sweep.py" --result_jsonl "${RESULTS}" "$@" || FAILED=1
else
    for IDX in "${!SWEEP_VALUES[@]}"; do
        VALUE="${SWEEP_VALUES[IDX]}"
        echo "--- point ${SWEEP_FLAG#--}${VALUE} ($(date +%H:%M:%S)) ---"
        # PROFILE=0：profiler 会扰动稳态读数，扫点一律关掉
        MASTER_PORT=$((BASE_PORT + IDX)) PROFILE=0 python -u "${REPO_ROOT}/examples/benchmark.py" \
            "${SWEEP_FLAG}" "${VALUE}" \
            "${EXTRA_ARGS[@]}" \
            --result_jsonl "${RESULTS}" \
            "$@" || FAILED=1
    done
fi

[ "${FAILED}" -ne 0 ] && { echo "有扫描点未跑完，不出图" >&2; exit 1; }

# 非 0 号 dp_rank 的节点不打印指标、不落 jsonl，跑完即正常退出
if [ ! -s "${RESULTS}" ]; then
    echo "本节点无 ${RESULTS}（dp_rank != 0），跳过出图"
    exit 0
fi

PLOT_ARGS=(--mode "${MODE}" --result_jsonl "${RESULTS}" --out "${OUT}/${MODE}_pareto.png")
[ "${MODE}" = "decode" ] && PLOT_ARGS+=(--draft_acceptance "${ACCEPT}")
python "${SCRIPT_DIR}/plot_pareto.py" "${PLOT_ARGS[@]}"
