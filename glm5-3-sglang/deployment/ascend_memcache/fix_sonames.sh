#!/usr/bin/env bash
# memcache-20261014 :: make the memfabric_hybrid native libraries resolvable by SONAME.
#
# WHY THIS IS NEEDED (measured, 2026-10-14):
#   memfabric_hybrid 1.1.5 ships its .so files in <site-packages>/memfabric_hybrid/lib
#   and <site-packages>/memfabric_hybrid.libs. Python's __init__ loads three of them by
#   absolute path, but hybm later does a *bare* dlopen("libhcom.so") for the host-RDMA
#   entity, and libhcom.so in turn NEEDs the auditwheel-mangled "libboundscheck-<hash>.so".
#   Those bare names are resolved only through the runtime loader search
#   (/lib, /usr/lib, LD_LIBRARY_PATH, caller RPATH) -- NOT through /etc/ld.so.conf.d,
#   which only feeds the ld.so *cache* and that cache is keyed by SONAME
#   (libhcom.so records SONAME libhcom.so.0, so a lookup of "libhcom.so" misses).
#   Setting LD_LIBRARY_PATH is NOT a usable workaround here: with LD_LIBRARY_PATH
#   non-empty and lacking the CANN paths, torch_npu fails to find libhccl.so.
# FIX: symlink every .so from both wheel dirs into /usr/lib (a default search dir).
# Container-local, reversible (rm the symlinks), original image untouched.
set -uo pipefail
SP=${1:-$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')}
LINKED=0
for d in "$SP/memfabric_hybrid/lib" "$SP/memfabric_hybrid.libs" "$SP/memcache_hybrid/lib" "$SP/memcache_hybrid.libs"; do
  [ -d "$d" ] || continue
  for f in "$d"/*.so*; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    if [ ! -e "/usr/lib/$base" ]; then
      ln -sf "$f" "/usr/lib/$base"
      echo "LINK /usr/lib/$base -> $f"
      LINKED=$((LINKED+1))
    fi
  done
done
echo "LINKED_COUNT=$LINKED"
ldconfig 2>/dev/null && echo "ldconfig done"
echo "--- resolution check ---"
for n in libhcom.so libboundscheck.so libmf_hybm_core.so libmf_smem.so libmf_memcache.so; do
  if [ -e "/usr/lib/$n" ]; then echo "  $n: present in /usr/lib"; else echo "  $n: MISSING"; fi
done
echo "OK"
