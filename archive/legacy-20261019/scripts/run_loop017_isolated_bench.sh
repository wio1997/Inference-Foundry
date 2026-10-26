#!/usr/bin/env bash
set -euo pipefail
root=/data/wio/Inference_Foundry
prefix=$(cat "$root/evidence/20260920_loop017_compressor/candidate_prefix.txt")
vendor=$prefix/vendors/custom_transformer
test -s "$vendor/op_api/lib/libcust_opapi.so"
test -f "$vendor/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/compressor.py"
docker exec \
  -e LOOP017_VENDOR="$vendor" \
  -e ASCEND_CUSTOM_OPP_PATH="$vendor" \
  -e LOOP017_ISOLATED_VENDOR=1 \
  -e LOOP017_REFERENCE="$root/artifacts/loop017_compressor_reference_output.pt" \
  -e LOOP017_OUT="$root/evidence/20260920_loop017_compressor/mbase256_screen.json" \
  dsv4ab bash -c '
    set -euo pipefail
    export LD_LIBRARY_PATH="$LOOP017_VENDOR/op_api/lib:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$LOOP017_VENDOR/op_api/lib/libcust_opapi.so${LD_PRELOAD:+:$LD_PRELOAD}"
    python3 /data/wio/Inference_Foundry/scripts/bench_loop017_compressor.py
  '
