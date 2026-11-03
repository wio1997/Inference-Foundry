#!/usr/bin/env python3
"""Metadata-only discovery in two fixed PROF roots; hash files, no CSV parse."""
import hashlib,json,os,re,stat
from pathlib import Path
base=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
samples=[('ep0','dp0_pp0_tp0_dcp0_ep0_rank0_728_20261028020238584_ascend_pt','PROF_000001_20261028020238598_00000728IKNPHKJQ','profiler_info_0.json','25c340bedc8129b4192aea6bb1569e6dab7c7eb77b31a9f738520dae2b58c50b','8d49b6a18fbd1f0c841345c4c9b4a6964dad5b2d624a508529c975ea6d96b6fc'),('ep8','dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt','PROF_000001_20261028020238610_00000727FHBEHQQL','profiler_info_8.json','a249b01a52dc6d9cbbd34953f2573b52d1d9f7280ff2d561b228b4a9417eb544','9196e0eecaff20235349999d756db25ed4a07995e2b652c2a23ad3945e43e09d')]
pat=re.compile(r'^op_summary_(?:slice_)?\d{1,20}.*\.csv');safe=re.compile(r'^[A-Za-z0-9_.-]{1,200}$')
def bounded_regular(p,maximum):
 if p.resolve(strict=True)!=p or not p.is_relative_to(base):raise RuntimeError('path boundary/symlink')
 fd=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode) or before.st_size>maximum:raise RuntimeError('file type/size')
  h=hashlib.sha256()
  while True:
   b=os.read(fd,1048576)
   if not b:break
   h.update(b)
  after=os.fstat(fd)
  if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('file changed')
  return {'sha256':h.hexdigest(),'size_bytes':before.st_size}
 finally:os.close(fd)
if base.resolve(strict=True)!=base:raise RuntimeError('base symlink')
results=[]
for label,rank,prof,info,info_sha,start_sha in samples:
 root=base/rank;profroot=root/prof
 a=bounded_regular(root/info,100000);b=bounded_regular(profroot/'host/start_info',100000)
 if a['sha256']!=info_sha or b['sha256']!=start_sha:raise RuntimeError('profile identity mismatch')
 dirs=[]
 for child in ('.','mindstudio_profiler_output','analyze'):
  d=profroot if child=='.' else profroot/child
  if not d.exists():dirs.append({'directory':child,'status':'missing','matches':[]});continue
  if d.resolve(strict=True)!=d or not d.is_relative_to(profroot) or not d.is_dir():raise RuntimeError('directory boundary/symlink')
  entries=[];count=0
  with os.scandir(d) as it:
   for e in it:
    count+=1
    if count>1000:raise RuntimeError('directory entry cap')
    if pat.fullmatch(e.name):
     if len(entries)>=30:raise RuntimeError('matched file cap')
     if not e.is_file(follow_symlinks=False):raise RuntimeError('match not regular file')
     q=bounded_regular(Path(e.path),268435456);visible=e.name if safe.fullmatch(e.name) else None
     entries.append({'filename':visible,'filename_sha256':None if visible else hashlib.sha256(e.name.encode()).hexdigest(),**q})
  dirs.append({'directory':child,'status':'read','entries_checked':count,'matches':sorted(entries,key=lambda x:x['filename'] or x['filename_sha256'])})
 if bounded_regular(root/info,100000)!=a or bounded_regular(profroot/'host/start_info',100000)!=b:raise RuntimeError('profile identity changed')
 results.append({'rank':label,'fixed_rank_directory':rank,'fixed_prof_directory':prof,'profiler_info_sha256':info_sha,'host_start_info_sha256':start_sha,'directories':dirs})
print(json.dumps({'scope':'Two fixed PROF roots, three immediate directories, anchored op_summary filename patterns; no CSV parsing','samples':results,'limitations':['Hash reads file bytes but never parses or emits file content.','Source file hashes are first observations and do not independently freeze data at capture.','Empty or missing selected directories do not exclude copies elsewhere or filenames outside regex.','Current parser source path rules do not prove the historical exporter version.','No NPU calls or user content.']},ensure_ascii=False,indent=2))
