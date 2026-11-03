#!/usr/bin/env python3
"""Read only three SHA-pinned profiler source files and fixed line windows."""
import hashlib,json,os,stat
from pathlib import Path

ITEMS=(
 ('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_view/_kernel_view_parser.py','d8c4c75c6ce95726726b20399cf05cd752be22d730ae49117ca9f2f4db0261fc',1,100),
 ('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis/common_func/data_manager.py','f02b37be762a8c9cbe31e3764ad6f7ffb7e497bea317825598b620c917c55221',65,95),
 ('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis/tuning/data_manager.py','0ee81d35a28e3fa60876f378d8409dc6eac64a50be811eceb2c3f806fecd039b',255,295),
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
                  'historical_binding':'current source hashes do not prove history runtime/execution',
                  'no_device_work':True},ensure_ascii=False,indent=2))
