#!/usr/bin/env python3
"""First-observed SHA and exact, AST-located methods from two fixed sources."""
import ast,hashlib,json,os,stat
from pathlib import Path
FILES=(
 ('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_bean/_common_bean.py','CommonBean',('__init__',)),
 ('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_common_func/_file_manager.py','FileManager',('read_csv_file','create_csv_file')),
)
results=[]
for filename,cls,required in FILES:
 p=Path(filename)
 if p.resolve(strict=True)!=p:raise RuntimeError('source path includes symlink')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=2_000_000:raise RuntimeError('invalid source type or size')
  with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(2_000_001)
  after=os.fstat(fd)
  if len(raw)!=before.st_size or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('source changed')
 finally:os.close(fd)
 text=raw.decode('utf-8');lines=text.splitlines();tree=ast.parse(text)
 classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls]
 if len(classes)!=1:raise RuntimeError('class missing or ambiguous')
 sections=[]
 for name in required:
  methods=[n for n in classes[0].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
  if len(methods)!=1:raise RuntimeError('method missing or ambiguous')
  n=methods[0];lo=min([n.lineno]+[d.lineno for d in n.decorator_list]);hi=n.end_lineno
  if hi is None or hi-lo+1>80:raise RuntimeError('method span exceeds limit')
  selected=lines[lo-1:hi]
  if any(len(line)>500 for line in selected):raise RuntimeError('line exceeds limit')
  sections.append({'method':cls+'.'+name,'first_line':lo,'last_line':hi,
                   'lines':[{'line':i,'text':line} for i,line in enumerate(selected,lo)]})
 results.append({'path':filename,'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'sections':sections})
print(json.dumps({'scope':'first-observed hashes and fixed class/method source only; no profile data',
 'files':results,'historical_binding':'current source does not prove historical runtime or loaded object',
 'no_device_work':True},ensure_ascii=False,indent=2))
