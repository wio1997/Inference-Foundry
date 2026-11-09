#!/usr/bin/env python3
"""Point-query archived C8 schema; never collect a profile or touch inference."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT=Path('audit/execution-search')
REMOTE=r'''
import sqlite3,json,pathlib,sys
s=json.loads(sys.argv[1]);p=pathlib.Path(s['database']);assert p.is_relative_to('/data/tiankuan/wio/')
st=p.stat();assert [st.st_size,st.st_mtime_ns]==s['identity']
c=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);c.execute('PRAGMA query_only=ON')
tables=[x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
strings=dict(c.execute('SELECT id,value FROM STRING_IDS'))
wanted={i:v for i,v in strings.items() if any(x in v.lower() for x in ['prepare input','draft_token','sample_token','allreduce','all_reduce','metadata','synchronize_input'])}
out=dict(database=str(p),tables=tables,matching_strings=wanted)
for t in ['PYTORCH_API','CANN_API','OSRT_API']:
 if t in tables:out[t]=list(c.execute('SELECT * FROM pragma_table_info(?)',(t,)))
c.close();after=p.stat();assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
print(json.dumps(out))
'''

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=OUT/'sampler-output-handoff-receipt.json'
    spec=json.loads(source.read_bytes())['source_specs'][0]
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=50)
    assert r.returncode==0,r.stderr
    d=json.loads(r.stdout);p=OUT/'metadata-host-profile-schema.json';p.write_text(r.stdout)
    receipt=dict(status='READ_ONLY_EXISTING_C8_SCHEMA_QUALIFICATION',source_receipt_sha256=digest(source),
        database=spec['database'],identity=spec['identity'],artifact=str(p),sha256=digest(p),reader_sha256=digest(__file__),
        table_count=len(d['tables']),Python_PyTorch_OSRT_tables=[t for t in d['tables'] if t in ('PYTORCH_API','OSRT_API')],
        new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
    (OUT/'metadata-host-profile-schema-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','table_count','Python_PyTorch_OSRT_tables','new_NPU_work')}))

if __name__=='__main__':main()
