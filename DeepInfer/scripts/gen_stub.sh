#!/bin/bash

set -ex

(
    CUDA_STUBS_DIR=$(python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(f"{CUDA_HOME}/lib64/stubs")')
    CUDA_STUB_LIBRARY="${CUDA_STUBS_DIR}/libcuda.so"
    if [ -e "${CUDA_STUB_LIBRARY}" ]; then
        export LD_PRELOAD="${CUDA_STUB_LIBRARY}"
    fi
    python -c 'import torch
from pybind11_stubgen import main
main(["deepinfer._C", "-o", "."])
'
)

# engine typing.Sequence -> Sequence
sed -i 's/typing.Sequence/Sequence/g' deepinfer/_C/engine.pyi
