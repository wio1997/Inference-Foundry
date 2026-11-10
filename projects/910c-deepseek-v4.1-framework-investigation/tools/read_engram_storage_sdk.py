#!/usr/bin/env python3
"""Snapshot installed CANN host registration declaration only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import OUT,REMOTE
if __name__=='__main__':
 catalog=json.loads((OUT/'engram-storage-catalog.json').read_text());row=catalog['CANN_header_catalog'][0];spec=[dict(runtime_path=row['path'],local_sha256=None)]
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(r.stderr)
 value=json.loads(r.stdout)
 for s in value['sources']:
  data=s.pop('content');assert sha256(data.encode()).hexdigest()==row['sha256']==s['runtime_sha256'];p=OUT/'engram-storage-runtime-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data);s['local_path']=str(p)
 value['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(OUT/'engram-storage-sdk.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(value['sources']),sha256=row['sha256'])))
