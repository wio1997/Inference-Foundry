#!/usr/bin/env python3
"""Read clock metadata from the two completed balanced profiler DBs, immutable read-only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,zlib
from read_cache_epilogue_balanced_calls import REMOTE

OUT=Path('audit/execution-search')
INNER=r'''
from pathlib import Path
from hashlib import sha256
import json,re,sqlite3
out=[]
for csvpath in paths:
 p=Path(csvpath);assert str(p).startswith('/data/tiankuan/wio/') and p.is_file() and p.parent.is_dir();dbs=sorted(p.parent.glob('*.db'));rows=[]
 for db in dbs:
  c=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True);tables=[r[0] for r in c.execute("select name from sqlite_master where type='table'")];sel=[n for n in tables if 'FREQ' in n.upper() or 'CLOCK' in n.upper()]
  hits=[]
  for n in sel:
   quoted='"'+n.replace('"','""')+'"';hits.append(dict(table=n,columns=[r[1] for r in c.execute('pragma table_info('+quoted+')')],rows=c.execute('select * from '+quoted+' limit 50').fetchall()))
  c.close()
  if hits:
   h=sha256()
   with db.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   rows.append(dict(path=str(db),bytes=db.stat().st_size,sha256=h.hexdigest(),clock_tables=hits))
 metadata=[]
 for info in sorted(p.parent.parent.rglob('info.json*')):
  if not info.is_file() or info.stat().st_size>1024*1024:continue
  raw=info.read_bytes()
  try:data=json.loads(raw)
  except (ValueError,UnicodeError):continue
  hits=[]
  def walk(x,prefix=''):
   if isinstance(x,dict):
    for key,value in x.items():
     path=prefix+'.'+key if prefix else key
     if 'freq' in key.lower():hits.append(dict(field=path,value=value))
     else:walk(value,path)
   elif isinstance(x,list):
    for j,value in enumerate(x):walk(value,prefix+'['+str(j)+']')
  walk(data)
  if hits:metadata.append(dict(path=str(info),sha256=sha256(raw).hexdigest(),bytes=len(raw),frequency_fields=hits))
 out.append(dict(csv_path=str(p),csv_exists=True,databases_found=len(dbs),clock_records=rows,info_json_frequency_metadata=metadata))
print(json.dumps(dict(saved_profile_clocks=out,scope='Existing completed profile DB clock metadata only; no live device queries or new measurements. SQLite mode=ro&immutable=1.')))
'''
if __name__=='__main__':
    rows=json.loads(zlib.decompress((OUT/'swa-publication-parent-calls.json.zlib').read_bytes()));paths=[r['path'] for r in rows]
    body='paths='+repr(paths)+'\n'+INNER
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,body,json.dumps(paths)],capture_output=True,text=True,timeout=120)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout);d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[]);(OUT/'o-projection-saved-clocks.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d['saved_profile_clocks'],ensure_ascii=False,indent=2))
