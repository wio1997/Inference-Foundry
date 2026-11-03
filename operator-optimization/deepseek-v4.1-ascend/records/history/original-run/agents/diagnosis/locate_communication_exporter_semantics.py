#!/usr/bin/env python3
"""Bounded read-only source inventory for current image communication DB tables."""
import hashlib,json,os,re
from pathlib import Path
roots=[Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis'),Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')]
pat=re.compile(r'\bCOMMUNICATION_OP\b|\bCOMMUNICATION_TASK_INFO\b|CommunicationOp|CommunicationTaskInfo',re.I)
results=[];scanned=0;skipped=[]
for root in roots:
 if not root.is_dir():skipped.append({'path':str(root),'reason':'missing root'});continue
 for base,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=sorted(d for d in dirs if d not in ('__pycache__','.git'))
  for name in sorted(files):
   if not name.endswith(('.py','.sql')):continue
   p=Path(base)/name
   try:
    raw=p.read_bytes()
    if len(raw)>2_000_000:skipped.append({'path':str(p),'reason':'over size cap'});continue
    s=raw.decode('utf8');scanned+=1
   except (OSError,UnicodeError) as e:skipped.append({'path':str(p),'reason':type(e).__name__});continue
   matched=[{'line':i,'text':line.strip()[:240]} for i,line in enumerate(s.splitlines(),1) if pat.search(line)]
   if matched:results.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'match_count':len(matched),'matches':matched[:15]})
   if len(results)>50:raise RuntimeError('too many matching source files')
print(json.dumps({'roots':[str(x) for x in roots],'files_scanned':scanned,'skipped':skipped,'matches':results,
 'limitations':['Only .py/.sql under these roots searched; compiled exporters and other paths or names are not covered.',
 'A reader join or a field name does not establish global identity, event pairing or cross-rank semantics.',
 'No input data or workload is executed.']},ensure_ascii=False,indent=2))
