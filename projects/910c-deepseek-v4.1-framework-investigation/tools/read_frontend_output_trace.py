#!/usr/bin/env python3
"""Read compact call intervals from the completed frontend trace, no new profile."""
from hashlib import sha256
import base64
import json
from pathlib import Path
import subprocess
import zlib
from read_inactive_dp_profile_probe import ROOT

REMOTE=r'''
from pathlib import Path
from hashlib import sha256
from collections import Counter
import base64,gzip,json,sys,zlib
root=Path(sys.argv[1]);assert root.is_relative_to('/data/tiankuan/wio')
p,=list((root/'traces').glob('*.pt.trace.json.gz'));st=p.stat();raw=p.read_bytes()
assert sha256(raw).hexdigest()=='44be01fb494800553d739417d8821e5aff9c5a9e28e3a927802c769155147cd2'
d=json.loads(gzip.decompress(raw));names=[];index={};rows=[];cats=Counter()
for e in d['traceEvents']:
 if e.get('ph')!='X':continue
 n=e.get('name','');cat=e.get('cat','');cats[cat]+=1
 if cat!='python_function':continue
 if n not in index:index[n]=len(names);names.append(n)
 a=e.get('args',{})
 rows.append([index[n],e['ts'],e.get('dur',0),e.get('pid'),e.get('tid'),a.get('Python id'),a.get('Python parent id')])
assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
meta={k:v for k,v in d.items() if k!='traceEvents'}
data=dict(source=dict(path=str(p),sha256=sha256(raw).hexdigest(),bytes=len(raw),mtime_ns=st.st_mtime_ns),metadata=meta,
 names=names,columns=['name_index','ts_us','dur_us','pid','tid','Python_id','Python_parent_id'],rows=rows,category_counts=dict(cats),
 scope='All complete Python-function call intervals only; durations/parent relations require audit before CPU-active or net-credit interpretation',
 new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
payload=json.dumps(data,separators=(',',':')).encode();packed=zlib.compress(payload,9)
print(json.dumps(dict(zlib_base64=base64.b64encode(packed).decode(),sha256=sha256(packed).hexdigest(),raw_bytes=len(payload))))
'''


def main():
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,ROOT],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    x=json.loads(r.stdout);packed=base64.b64decode(x.pop('zlib_base64'));assert sha256(packed).hexdigest()==x['sha256']
    d=json.loads(zlib.decompress(packed));out=Path('audit/execution-search/frontend-output-events.json.zlib');out.write_bytes(packed)
    receipt=dict(path=str(out),**x,compressed_bytes=len(packed),source=d['source'],rows=len(d['rows']),names=len(d['names']),
                 category_counts=d['category_counts'],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                 new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
    Path('audit/execution-search/frontend-output-trace-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
