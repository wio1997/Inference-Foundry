#!/usr/bin/env python3
"""Inside the dedicated container: bounded, read-only lookup under one source subtree."""
import hashlib, json, os, re
from pathlib import Path
root=Path('/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine')
names={'moe_token_unpermute.h','hccl_shmem.hpp'}
patterns={
 'moe_token_unpermute.h':re.compile(r'Process\s*\(|expanded|permut|prob|round|cast|scale|accum|DataCopy|Sync|Gather',re.I),
 'hccl_shmem.hpp':re.compile(r'CrossRankSync|sync|signal|wait|barrier|fence|DataCopy|mem',re.I),
}
out={'root':str(root),'target_names':sorted(names),'files':[],'scan_complete':True,'entries_seen':0}
if root.is_symlink() or not root.is_dir():
 out.update(scan_complete=False,reason='root missing, not directory, or symlink')
else:
 def onerror(error):
  out.update(scan_complete=False,reason='directory read error')
 for directory, dirs, files in os.walk(root,topdown=True,followlinks=False,onerror=onerror):
  if not out['scan_complete']: break
  symlinks=[d for d in dirs if (Path(directory)/d).is_symlink()]
  if symlinks: out.setdefault('skipped_symlink_dirs',[]).extend(str(Path(directory)/d) for d in symlinks)
  dirs[:]=sorted(d for d in dirs if d not in symlinks)
  files.sort()
  out['entries_seen']+=len(dirs)+len(files)
  if out['entries_seen']>20000:
   out.update(scan_complete=False,reason='entry limit exceeded'); break
  for name in files:
   if name not in names: continue
   p=Path(directory)/name
   if p.is_symlink() or not p.is_file():
    out['files'].append({'path':str(p),'status':'symlink_or_nonregular'}); continue
   if len(out['files'])>=16:
    out.update(scan_complete=False,reason='match limit exceeded'); break
   n=p.stat().st_size
   rec={'path':str(p),'size_bytes':n}
   if n>1048576:
    rec['status']='over_1MiB_not_read';out['files'].append(rec); continue
   with p.open('rb') as f: b=f.read(1048577)
   if len(b)>1048576 or len(b)!=n:
    rec['status']='size_changed_or_limit_exceeded';out['files'].append(rec);continue
   rec.update(status='read',sha256=hashlib.sha256(b).hexdigest())
   try: lines=b.decode('utf-8').splitlines()
   except UnicodeDecodeError:
    rec['status']='non_utf8';out['files'].append(rec);continue
   hits=[i for i,l in enumerate(lines) if patterns[name].search(l)]
   selected=sorted({j for i in hits for j in range(max(0,i-2),min(len(lines),i+3))})
   rec['line_count']=len(lines);rec['hit_count']=len(hits);rec['selected_line_count']=len(selected)
   rec['snippet_truncated']=len(selected)>200 or any(len(lines[i])>512 for i in selected[:200])
   rec['snippet']=[{'line':i+1,'text':lines[i][:512]} for i in selected[:200]]
   out['files'].append(rec)
  if not out['scan_complete']: break
if out.get('skipped_symlink_dirs'):
 out.update(scan_complete=False,reason='symlink directories skipped')
out['missing_names'] = sorted(names - {Path(f['path']).name for f in out['files']}) if out['scan_complete'] else []
print(json.dumps(out,ensure_ascii=False))
