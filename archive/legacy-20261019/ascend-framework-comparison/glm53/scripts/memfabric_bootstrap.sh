#!/bin/bash
# Project-private aliases for native dlopen names; preserve the full image env.
set -eu
glm53_alias_dir=/tmp/glm53-sdk-alias
glm53_python_packages=/usr/local/python3.12.13/lib/python3.12/site-packages
mkdir -m 700 "$glm53_alias_dir"
ln -s "$glm53_python_packages/memfabric_hybrid.libs/libmf_hybm_core-00225bf3.so" "$glm53_alias_dir/libmf_hybm_core.so"
ln -s "$glm53_python_packages/memfabric_hybrid.libs/libmf_smem-2154e855.so" "$glm53_alias_dir/libmf_smem.so"
ln -s "$glm53_python_packages/memfabric_hybrid.libs/libmf_acc_offload-0c99b633.so" "$glm53_alias_dir/libmf_acc_offload.so"
ln -s "$glm53_python_packages/memfabric_hybrid.libs/libboundscheck-58fadf91.so" "$glm53_alias_dir/libboundscheck.so"
export LD_LIBRARY_PATH="$glm53_alias_dir:$glm53_python_packages/memcache_hybrid/lib:$glm53_python_packages/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
exec python3 -c "$1" "$2" "$3" "$4"
