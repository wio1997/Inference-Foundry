#!/usr/bin/env bash
# Runs inside a dedicated ARM64 build container, without NPU device passthrough.
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive DEEPINFER_ACCELERATOR=ascend
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0 ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0
export MAX_JOBS=${MAX_JOBS:-16} PYTHONUNBUFFERED=1
cd /work
test -f locks/requirements-engine.txt
test -f src/DeepInfer/requirements-dev.txt

if ! command -v g++-13 >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends curl ca-certificates gnupg
  curl -4 -fLsS 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC8EC952E2A0E1FBDC5090F6A2C277A0A352154E5' \
    -o /usr/share/keyrings/ubuntu-toolchain-r.asc
  printf '%s\n' 'deb [signed-by=/usr/share/keyrings/ubuntu-toolchain-r.asc] https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu jammy main' \
    > /etc/apt/sources.list.d/deepinfer-toolchain.list
  apt-get update
  apt-get install -y --no-install-recommends g++-13
fi
apt-get update
apt-get install -y --no-install-recommends git libdw-dev libnuma-dev
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13

python3 apply_patches.py
python3 -m venv /work/engine-venv
/work/engine-venv/bin/python -m pip install --no-index --find-links /work/wheelhouse \
  -r /work/locks/requirements-engine.txt
source /work/engine-venv/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}"
set +u
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -u
python /work/configure_compilers.py
mkdir -p /work/wheels
for package in DeepSelect FlashMLA DeepEP-Ascend DeepGEMM-Ascend; do
  (cd "/work/src/DeepInfer/vendor/$package"
   DEEP_SELECT_BUILD_TARGET_PLATFORM=ASCEND FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND \
     python setup.py bdist_wheel --dist-dir /work/wheels)
done
python -m pip install --no-deps --no-index --find-links /work/wheels deep_select flash_mla deep_ep deep_gemm
(cd /work/src/DeepInfer && bash scripts/build.sh build)
python -m pip freeze > /work/engine-freeze.rebuilt.txt

# Protocol adapter: a separate CPU-only Python environment, same versions as the validated service.
python3 -m venv /work/recipe-venv
/work/recipe-venv/bin/python -m pip install --no-index --find-links /work/wheelhouse \
  'deepseek-recipe==0.2.0' 'fastapi==0.128.0' 'uvicorn==0.40.0'
printf '%s\n' BUILD_COMPLETE_DEVICE_TEST_PENDING > /work/build-state.txt
echo 'BUILD_COMPLETE_DEVICE_TEST_PENDING (device correctness checks are still required)'
