#!/bin/bash
# Development setup: builds extension and symlinks into package directory

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old build artifacts
rm -rf build dist

# Build
DG_USE_LOCAL_VERSION=0 python3 setup.py build

# Find the .so file in build directory and create symlink in package directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sf "$(realpath "$so_file")" deep_gemm/
    ln -sf ../stubs/_C.pyi deep_gemm
    echo "Linked: $so_file -> deep_gemm/"
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Return to original directory
cd "$original_dir"
