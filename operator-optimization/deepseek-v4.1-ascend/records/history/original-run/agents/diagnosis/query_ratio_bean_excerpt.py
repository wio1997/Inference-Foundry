#!/usr/bin/env python3
"""Read only two first-observed SHA-pinned profiler source files and fixed line windows."""
import hashlib,json,os,stat
from pathlib import Path

ITEMS=(
 ('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_bean/_op_summary_bean.py','982d91fec90085fed7a5a17f35d987292fb3bfac531c60a356d0116d64a328f6',1,50),
 ('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_common_func/_csv_headers.py','651a3702d4cc32c87e166280d4f0cadfdb9fabaa81a948a0f6533fc3c0121947',1,35),
)
results=[]
for filename,digest,lo,hi in ITEMS:
    p=Path(filename)
    if p.resolve(strict=True)!=p: raise RuntimeError('source path includes symlink')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or not 0<a.st_size<=2_000_000:raise RuntimeError('invalid file type or size')
        with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(2_000_001)
        b=os.fstat(fd)
        if len(raw)!=a.st_size or (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)!=(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns):raise RuntimeError('file changed')
    finally:os.close(fd)
    if hashlib.sha256(raw).hexdigest()!=digest:raise RuntimeError('inventory hash mismatch')
    lines=raw.decode('utf-8').splitlines()
    if len(lines)<lo:raise RuntimeError('range does not exist')
    selected=lines[lo-1:hi]
    if any(len(x)>500 for x in selected):raise RuntimeError('line exceeds limit')
    results.append({'path':filename,'sha256':digest,'first_line':lo,'last_line':min(hi,len(lines)),
                    'lines':[{'line':n,'text':text} for n,text in enumerate(selected,lo)]})
print(json.dumps({'scope':'fixed source line ranges only; no profile data', 'files':results,
                  'historical_binding':'stage270 hashes first observed in current container; no proof of history runtime/execution',
                  'no_device_work':True},ensure_ascii=False,indent=2))
