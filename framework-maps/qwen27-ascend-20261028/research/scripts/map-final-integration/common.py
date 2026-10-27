from pathlib import Path
import json,os,hashlib

def atomic(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with q.open('x')as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def rec(p):
 p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
