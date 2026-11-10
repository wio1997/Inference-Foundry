#!/usr/bin/env python3
"""Recheck actual CP/base/frame owners without a process inside the container."""
from pathlib import Path
from hashlib import sha256
import json, subprocess
from read_swa_query_producer_entry import REMOTE

OUT=Path('audit/execution-search')
def main():
    prior=json.loads((OUT/'common-metadata-dependencies.json').read_text())['sources']
    names=['worker/model_runner_v1.py','attention/context_parallel/dsa_v41_cp.py',
           'attention/context_parallel/dsa_cp.py','attention/dsa_v41.py','worker/device_metadata.py']
    specs=[]
    for n in names:
        r=next(r for r in prior if r['runtime_path'].endswith('/'+n));p=Path(r['local_path'])
        specs.append(dict(runtime_path=r['runtime_path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(specs)],capture_output=True,text=True,timeout=110)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for row in d['sources']:
        text=row.pop('content',None)
        if text is not None:
            p=OUT/'common-frame-measured-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
            assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk source identities; functiontiming is separate olddiagnostic and not zerooverhead/primitivebit proof.')
    (OUT/'common-frame-measured-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),container_state=d['container_state'],new_NPU_work=0)))
if __name__=='__main__':main()
