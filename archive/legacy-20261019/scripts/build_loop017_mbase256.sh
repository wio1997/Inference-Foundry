#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
src=/vllm-workspace/vllm-ascend
root=/data/wio/Inference_Foundry
tag=loop017_compressor_mbase256_$(date -u +%Y%m%dT%H%M%SZ)
stage=/tmp/$tag
prefix=$root/artifacts/$tag
test ! -e "$stage"
test ! -e "$prefix"
mkdir -p "$stage" "$prefix" "$root/evidence/20260920_loop017_compressor"
printf '%s\n' "$prefix" > "$root/evidence/20260920_loop017_compressor/candidate_prefix.txt"
tar -C "$src" --exclude=csrc/build --exclude=csrc/output --exclude=csrc/build_out -cf - csrc | tar -C "$stage" -xf -
tiling=$stage/csrc/attention/compressor/op_host/arch32/compressor_tiling.cpp
python3 - "$tiling" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()
old="innerSplitParams_->mBaseSize = 128;"
assert s.count(old)==1, s.count(old)
p.write_text(s.replace(old,"innerSplitParams_->mBaseSize = 256;"))
PY
diff -u "$src/csrc/attention/compressor/op_host/arch32/compressor_tiling.cpp" "$tiling" > "$root/patches/loop017_compressor_mbase256.patch" || test $? -eq 1
cd "$stage/csrc"
bash build.sh --pkg --ops=compressor --soc=ascend910b --vendor_name=custom -j8
shopt -s nullglob
packages=(build/cann-ops-transformer*.run)
test "${#packages[@]}" -eq 1
"${packages[0]}" --install-path="$prefix" --quiet
vendor=$prefix/vendors/custom_transformer
test -s "$vendor/op_api/lib/libcust_opapi.so"
test -f "$vendor/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/compressor.py"
printf 'PREFIX=%s\nVENDOR=%s\n' "$prefix" "$vendor"
