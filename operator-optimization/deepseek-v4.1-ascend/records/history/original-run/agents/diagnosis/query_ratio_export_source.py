#!/usr/bin/env python3
"""Read-only fixed-word scan of two container profiler source roots. No data files."""
import hashlib
import json
import os
import re
import stat
from pathlib import Path

ROOTS = (
    Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis'),
    Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis'),
)
WORDS = ('aic_mac_ratio', 'aiv_vec_ratio', 'aic_total_cycles', 'aiv_total_cycles', 'kernel_details')
PATTERN = re.compile('|'.join(map(re.escape, WORDS)), re.IGNORECASE)
MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_HITS = 2000, 2_000_000, 200_000_000, 1000
results, files_scanned, bytes_scanned, hit_count = [], 0, 0, 0
skips = {'symlink_dirs': 0, 'symlink_files': 0, 'nonregular_files': 0, 'oversize_files': 0, 'non_utf8_files': 0}
for root in ROOTS:
    if root.resolve(strict=True) != root or not root.is_dir():
        raise RuntimeError('root missing, not a directory, or includes symlink')
    for base, dirs, names in os.walk(root, topdown=True, followlinks=False):
        if len(Path(base).relative_to(root).parts) > 32:
            raise RuntimeError('directory depth limit')
        kept=[]
        for d in sorted(dirs):
            path=Path(base)/d
            if path.is_symlink(): skips['symlink_dirs']+=1
            elif d not in ('__pycache__','.git'): kept.append(d)
        dirs[:]=kept
        for name in sorted(names):
            if not name.endswith(('.py','.sql')): continue
            path=Path(base)/name
            if path.is_symlink(): skips['symlink_files']+=1; continue
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode): skips['nonregular_files']+=1; continue
                if before.st_size>MAX_FILE_BYTES: skips['oversize_files']+=1; continue
                files_scanned+=1
                if files_scanned>MAX_FILES: raise RuntimeError('file count limit')
                bytes_scanned+=before.st_size
                if bytes_scanned>MAX_TOTAL_BYTES: raise RuntimeError('byte limit')
                with os.fdopen(fd,'rb',closefd=False) as f: raw=f.read(MAX_FILE_BYTES+1)
                after=os.fstat(fd)
                if len(raw)!=before.st_size or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                    raise RuntimeError('source file changed')
            finally:
                os.close(fd)
            try: lines=raw.decode('utf-8').splitlines()
            except UnicodeDecodeError: skips['non_utf8_files']+=1; continue
            matches=[]
            for line_no,line in enumerate(lines,1):
                found=sorted({m.group(0).lower() for m in PATTERN.finditer(line)})
                if found:
                    hit_count+=1
                    if hit_count>MAX_HITS: raise RuntimeError('hit count limit')
                    matches.append({'line':line_no,'keywords':found})
            if matches:
                results.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'matches':matches})
print(json.dumps({'roots':list(map(str,ROOTS)), 'extensions':['.py','.sql'], 'keywords':list(WORDS),
                  'files_scanned':files_scanned,'bytes_scanned':bytes_scanned,'skips':skips,'hit_count':hit_count,
                  'results':results,'scope':'current container text source filenames, hashes and keyword line numbers only; no code or profile data; compiled files excluded',
                  'historical_binding':'same fixed image and clean mount inspection do not prove historical loaded objects or files unchanged',
                  'no_device_work':True},ensure_ascii=False,indent=2))
