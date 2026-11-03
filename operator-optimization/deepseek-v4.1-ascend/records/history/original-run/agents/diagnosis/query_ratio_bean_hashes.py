#!/usr/bin/env python3
"""Discover hashes only for the two imports in the SHA-pinned kernel parser."""
import hashlib,json,os,stat
from pathlib import Path
FILES=(
 '/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_bean/_op_summary_bean.py',
 '/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_common_func/_csv_headers.py',
)
rows=[]
for path in map(Path,FILES):
 if path.resolve(strict=True)!=path:raise RuntimeError('source path includes symlink')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=2_000_000:raise RuntimeError('source type or size')
  with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(2_000_001)
  after=os.fstat(fd)
  if len(raw)!=before.st_size or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('source changed')
 finally:os.close(fd)
 rows.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)})
print(json.dumps({'files':rows,'scope':'fixed source file hashes only; no contents or profile data',
 'historical_binding':'current hashes do not prove history runtime or loaded objects','no_device_work':True},indent=2))
