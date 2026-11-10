#!/usr/bin/env python3
"""Read all saved native task rows for Q/KV publication dependency binding."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,zlib
from read_cache_epilogue_balanced_calls import REMOTE

INNER=r'''
from pathlib import Path
from hashlib import sha256
import csv,json,sys
out=[]
for spec in json.loads(sys.argv[1]):
 p=Path(spec['path']);assert p.is_relative_to('/data/tiankuan/wio') and p.name=='kernel_details.csv';raw=p.read_bytes();rows=list(csv.DictReader(raw.decode().splitlines()));assert rows
 fields=list(rows[0]);selected=[]
 for i,r in enumerate(rows):
  selected.append(dict(csv_row=i+2,data={k:v for k,v in r.items() if k in ['Name','Type','Stream ID','Task ID','Task Type','Start Time(us)','Duration(us)','Input Shapes','Input Data Types','Input Formats','Output Shapes','Output Data Types','Output Formats','Block Num','Mix Block Num','aicore_time(us)','aiv_time(us)']}))
 out.append(dict(**spec,sha256=sha256(raw).hexdigest(),total_csv_rows=len(rows),headers=fields,selected=selected))
print(json.dumps(out))
'''
if __name__=='__main__':
 root=Path('audit/execution-search');inv=json.loads(Path('audit/balanced-profile-171/kernel-full-inventory.json').read_text());spec=[{k:r[k] for k in ['phase','ep','dp','path']} for r in inv['files'] if r['phase']=='steady' and r['ep'] in [0,8]];assert len(spec)==2
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,INNER,json.dumps(spec)],capture_output=True,text=True,timeout=100)
 if r.returncode:raise RuntimeError(r.stderr)
 data=json.loads(r.stdout);raw=json.dumps(data,separators=(',',':')).encode();dest=root/'swa-publication-parent-calls.json.zlib';dest.write_bytes(zlib.compress(raw,9))
 meta=dict(profiles=[{k:v for k,v in d.items() if k!='selected'}|dict(retained_rows=len(d['selected'])) for d in data],archive=dict(path=str(dest),sha256=sha256(dest.read_bytes()).hexdigest(),uncompressed_sha256=sha256(raw).hexdigest()),reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='All task rows from onlyEP0/EP8steady saved kernel CSVs; representative native geometry/dependency entry, notallranks/fullrequests orCPU/runtimeevent proof.',new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[])
 (root/'swa-publication-parent-call-pins.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(profiles=len(data),retained_rows=[len(d['selected']) for d in data],new_profile=0)))
