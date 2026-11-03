#!/usr/bin/env python3
"""Read-only bounded search in frozen image profiler source roots; stdout only."""
import hashlib,json,os,re
from pathlib import Path
roots=[Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis'),
 Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')]
pattern=re.compile(r'TASK_PMU_INFO|TaskPmuInfo|task_pmu_info|globalTaskId',re.I)
results=[];scanned=0;skipped=[]
for root in roots:
 if not root.is_dir():skipped.append(str(root));continue
 for base,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git'))
  for name in sorted(files):
   if not name.endswith(('.py','.sql')):continue
   p=Path(base)/name
   try:
    raw=p.read_bytes()
    if len(raw)>2_000_000:skipped.append(str(p)+' (large)');continue
    s=raw.decode('utf-8');scanned+=1
   except (OSError,UnicodeError):continue
   matches=[{'line':n,'text':line.strip()[:240]} for n,line in enumerate(s.splitlines(),1) if pattern.search(line)]
   if matches:results.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'matches':matches[:12], 'match_count':len(matches)})
if len(results)>40:raise RuntimeError('Too many matches; narrow search explicitly')
print(json.dumps({'roots':list(map(str,roots)),'files_scanned':scanned,'skipped':skipped,'results':results,
'scope':'Python/SQL text only; absence does not exclude compiled CANN exporter or other versions'},ensure_ascii=False,indent=2))
