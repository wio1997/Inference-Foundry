#!/usr/bin/env bash
# warm93-memcache-aisbench-20261015 :: DECODE rank launcher (NO MemCache / NO HiCache / NO L3).
#
# Global TP32 / DP16 / EP32, attention TP2 across two hosts, eager, native Ascend PD.
# User requirement: memory fraction 0.90 (0.80 was explicitly rejected).
# Args: $1 = node (36|41). Runs INSIDE the owned container sglang-nightly-nomc-d-<N>-20261015.
set -u
N=${1:?node}
ROOT=${REMOTE_ROOT:-/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015}
D=$ROOT
cd "$D" || exit 3
case "$N" in
  36) RANK=0 ;;
  41) RANK=1 ;;
  *) echo "outside D group"; exit 2 ;;
esac
NEED_PORTS="31401"
[ "$N" = 36 ] && NEED_PORTS="$NEED_PORTS 29836"
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

# Dedicated decode configuration: low_latency DeepEP, isolate 650MB MoE from 64MB non-MoE.
# The P long-sequence normal window is not needed by this D-only process group.
unset DEEPEP_NORMAL_LONG_SEQ_ROUND DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ
export HCCL_BUFFSIZE=64
export DEEPEP_HCCL_BUFFSIZE=650
export TASK_QUEUE_ENABLE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

exec python3 -m sglang.launch_server \
 --enable-metrics --enable-cache-report \
 --model-path /model --trust-remote-code --device npu --attention-backend ascend \
 --dtype bfloat16 --quantization modelslim \
 --tp-size 32 --dp-size 16 --ep-size 32 --enable-dp-attention --enable-dp-lm-head \
 --nnodes 2 --node-rank "$RANK" --dist-init-addr 172.16.12.36:29836 --dist-timeout 3600 \
 --host "172.16.12.$N" --port 31401 --served-model-name glm-5.3-pd \
 --disaggregation-mode decode --disaggregation-transfer-backend ascend \
 --context-length 98304 --mem-fraction-static 0.90 \
 --max-running-requests 16 \
 --moe-a2a-backend deepep --deepep-mode low_latency --disable-cuda-graph \
 --disable-shared-experts-fusion \
 --reasoning-parser glm45 --watchdog-timeout 3600
