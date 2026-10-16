#!/usr/bin/env bash
# warm93-memcache-aisbench-20261015 :: PREFILL rank launcher (official1.3 native MemCache profile).
#
# Global TP64 / DP4 / EP64, attention TP16, optimized, native Ascend PD over MemFabric.
# Args: $1 = node (33|34|35|37). Runs INSIDE the owned container sglang-nightly-nomc-p-<N>-20261015.
set -u
N=${1:?node}
ROOT=${REMOTE_ROOT:-/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015}
D=$ROOT
cd "$D" || exit 3
case "$N" in
  34) RANK=0 ;;
  33) RANK=1 ;;
  35) RANK=2 ;;
  37) RANK=3 ;;
  *) echo "outside P group"; exit 2 ;;
esac
NEED_PORTS="31400 31934"
[ "$N" = 34 ] && NEED_PORTS="$NEED_PORTS 29834"
python3 - "$N" $NEED_PORTS <<'PORTS'
import socket,sys
host='172.16.12.'+sys.argv[1]
for port in map(int,sys.argv[2:]):
 with socket.socket() as s:
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try: s.bind((host,port))
  except OSError as e: print('PORT_BIND_CHECK_FAILED',host,port,str(e));sys.exit(5)
print('PORT_BIND_CHECK_PASS',host,sys.argv[2:])
PORTS
[ "$?" = 0 ] || exit 5
[ -f "$D/scripts/warm-common-env.sh" ] || { echo "ABORT: warm-common-env.sh missing"; exit 5; }
# shellcheck disable=SC1091
source "$D/scripts/warm-common-env.sh" || { echo "ABORT: common environment failed"; exit 5; }

# ---- communication settings preserved from the verified 4P2D baseline ----
export HCCL_BUFFSIZE=1000
export TASK_QUEUE_ENABLE=2
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export SGLANG_NPU_MEMCACHE_ENABLE_WARMUP=1
MC_EXTRA=$(cat "$ROOT/conf/model-memcache.json")
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=0
exec python3 -m sglang.launch_server \
 --model-path /model --trust-remote-code --device npu --attention-backend ascend \
 --dtype bfloat16 --quantization modelslim \
 --tp-size 64  --ep-size 64 --attn-dp-size 4 --enable-dp-lm-head \
 --nnodes 4 --node-rank "$RANK" --dist-init-addr 172.16.12.34:29834 --dist-timeout 3600 \
 --host "172.16.12.$N" --port 31400 --served-model-name glm-5.3-pd \
 --disaggregation-mode prefill --disaggregation-bootstrap-port 31934 \
 --disaggregation-transfer-backend ascend \
 --context-length 98304 --mem-fraction-static 0.75 \
 --enable-metrics --enable-cache-report \
 --enable-hierarchical-cache --hicache-io-backend kernel_ascend \
 --hicache-storage-backend npu_memcache --hicache-mem-layout page_first_kv_split \
 --hicache-size 32 --hicache-host-memory-mode buffer_only --hicache-write-policy write_through \
 --hicache-storage-prefetch-policy wait_complete \
 --hicache-storage-backend-extra-config "$MC_EXTRA" \
 --chunked-prefill-size 16384 --max-prefill-tokens 98304 \
 --max-running-requests 16 \
 --moe-a2a-backend deepep --deepep-mode normal --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
 --disable-shared-experts-fusion --moe-dense-tp-size 1 \
 --tokenizer-worker-num 4 --load-balance-method round_robin \
 --speculative-algorithm NEXTN --speculative-eagle-topk 1 \
 --speculative-draft-model-quantization modelslim \
 --speculative-num-steps 1 --speculative-num-draft-tokens 2 \
 --reasoning-parser glm45 --watchdog-timeout 3600
