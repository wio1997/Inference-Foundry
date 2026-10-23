#!/usr/bin/env bash
# Run inside the new deployment container.
set -euo pipefail
cd /workspace/deploy
source ./cann_env.sh
test "$(git -C /vllm-workspace/vllm rev-parse HEAD)" = 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
test "$(git -C /opt/vllm-ascend-dsnext rev-parse HEAD)" = 2c00c27e5fa1c5458720a516a8b7897f0b2dc7ac
python3 ./patch_dsa.py
SRC=/workspace/deploy/src/cann-recipes-infer
PIN=6d060311ae75f6105fe4216e20566875da5a8838
mkdir -p /workspace/deploy/src /workspace/deploy/opp /workspace/deploy/artifacts
if [ ! -e "$SRC" ]; then
  git clone https://gitcode.com/cann/cann-recipes-infer.git "$SRC"
  git -C "$SRC" checkout --detach "$PIN"
fi
test "$(git -C "$SRC" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$SRC" status --porcelain --untracked-files=no)"
cd "$SRC/ops/ascendc"
OPS_CPU_NUMBER=16 bash build.sh -c ascend950 -n moe_gating_top_k_hash \
  2>&1 | tee /workspace/deploy/logs/build-moe-routing.log
PACKAGE="$SRC/ops/ascendc/build/CANN-custom_ops-none-linux.aarch64.run"
test -f "$PACKAGE"
UNPACK="/workspace/deploy/artifacts/moe-routing-$(date -u +%Y%m%dT%H%M%SZ)"
bash "$PACKAGE" --noexec --extract="$UNPACK"
cd "$UNPACK"
bash ./install.sh --install-path=/workspace/deploy/opp --quiet
PROVIDER=/workspace/deploy/opp/vendors/customize
test -f "$PROVIDER/op_api/lib/libcust_opapi.so"
nm -D "$PROVIDER/op_api/lib/libcust_opapi.so" | grep -E 'aclnnMoeGatingTopKHash(GetWorkspaceSize)?$'
sha256sum "$PACKAGE" "$PROVIDER/op_api/lib/libcust_opapi.so" \
  "$PROVIDER/op_api/include/aclnn_moe_gating_top_k_hash.h" \
  > /workspace/deploy/artifacts/operator-sha256.txt
echo 'Preparation complete. Run the routing smoke before starting the model.'
