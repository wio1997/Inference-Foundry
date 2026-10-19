"""Archive local immutable collector JSON; keep Git references and file digests."""
from pathlib import Path
import hashlib,json,subprocess,tarfile
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
backup=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/large-collector-json-v1');backup.mkdir(exist_ok=True)
paths=[repo/s for s in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines() if s.startswith(str(r.relative_to(repo))+'/')]
large=[p for p in paths if p.stat().st_size>8*1024**2]
rows=[]
for root in sorted(set(p.parent for p in large)):
 members=sorted(p for p in root.iterdir() if p.is_file());dest=backup/(root.name+'.tar.gz')
 if not dest.exists():
  tmp=dest.with_suffix('.partial')
  with tarfile.open(tmp,'w:gz',compresslevel=3) as tar:
   for p in members:tar.add(p,arcname=p.name,recursive=False)
  tmp.rename(dest)
 with tarfile.open(dest,'r:gz') as tar:
  for p in members:assert hashlib.sha256(tar.extractfile(p.name).read()).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest()
 rows.append({'run':str(root.relative_to(repo)),'local_archive':str(dest),'archive_bytes':dest.stat().st_size,'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'all_members_verified':True,'Git_omitted_large_members':[{'path':str(p.relative_to(repo)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in large if p.parent==root]})
existing=json.loads((r/'large-collector-json-backup-v1.json').read_text()).get('runs',[]) if (r/'large-collector-json-backup-v1.json').exists() else []
merged={x['run']:x for x in existing}
merged.update({x['run']:x for x in rows})
rows=list(merged.values())
(r/'large-collector-json-backup-v1.json').write_text(json.dumps({'completed':True,'runs':rows,'scope':'Local backup of already frozen downloaded collector files. Large raw metrics/stdout omitted from Git; no node work or hardware requests.'},indent=2)+'\n')
exclude=repo/'.git/info/exclude'
if not exclude.parent.is_dir():
 exclude=Path(subprocess.check_output(['git','rev-parse','--git-path','info/exclude'],cwd=repo,text=True).strip());exclude=exclude if exclude.is_absolute() else repo/exclude
old=exclude.read_text() if exclude.exists() else ''
new=''.join('/'+str(p.relative_to(repo))+'\n' for p in large if '/'+str(p.relative_to(repo))+'\n' not in old)
with exclude.open('a') as f:f.write(new)
print(json.dumps({'archives':len(rows),'large_raw_files':len(large),'total_raw_bytes':sum(p.stat().st_size for p in large)}))
