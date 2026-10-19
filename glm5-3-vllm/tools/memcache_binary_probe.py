#!/usr/bin/env python3
"""Inspect MemCache ELF metadata without initializing an NPU or a service."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def run(argv, limit=16000):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    return {"rc": p.returncode, "stdout": p.stdout[:limit], "stderr": p.stderr[:2000]}


def main():
    tools = {n: shutil.which(n) for n in ("gdb", "readelf", "nm", "c++filt", "gcc", "g++", "cmake", "ninja")}
    result = {"tools": tools, "libraries": [], "packages": []}
    base = Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
    for name in ('memcache_hybrid', 'memfabric_hybrid'):
        package = base/name
        result['packages'].append({'path': str(package), 'version': (package/'VERSION').read_text()})
        for lib in sorted(package.rglob('*.so*')):
            if not lib.is_file():
                continue
            data = lib.read_bytes()
            rec = {'path': str(lib), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            if tools['readelf']:
                dynamic = run([tools['readelf'], '-d', str(lib)])
                rec['dynamic'] = [s.strip() for s in dynamic['stdout'].splitlines()
                                  if re.search(r'NEEDED|RPATH|RUNPATH|SONAME|SYMBOLIC', s)]
            if tools['nm']:
                p = subprocess.run([tools['nm'], '-D', '-C', str(lib)], capture_output=True, text=True, timeout=60)
                lines = p.stdout.splitlines()
                rec['symbol_counts'] = {s: sum(s in v for v in lines) for s in
                                        ('std::__cxx11', 'ock::acc::', 'ock::smem::', 'spdlog::', 'MfTlsUtil')}
                rec['symbols'] = [v for v in lines if re.search(
                    r'AccCommonServerOptions|AccTcpServerOptions|smem_get_abi_version|MfTlsUtil|basic_string.*assign', v)][:35]
            result['libraries'].append(rec)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
