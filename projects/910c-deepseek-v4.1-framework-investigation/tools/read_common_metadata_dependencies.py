#!/usr/bin/env python3
"""Recheck actual common metadata sources without importing serving frameworks."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE

OUT=Path('audit/execution-search')
def main():
    prior=json.loads((OUT/'metadata-readers-execution-identity.json').read_text())['sources']
    names=['worker/model_runner_v1.py','attention/context_parallel/dsa_v41_cp.py','attention/context_parallel/dsa_cp.py','attention/dsa_v41.py','attention/dsa_v1.py','worker/device_metadata.py','ascend_forward_context.py','utils.py','models/deepseek_v41/indexer.py']
    spec=[]
    for name in names:
        r=next(r for r in prior if r['runtime_path'].endswith('/'+name));p=Path(r['local_path'])
        spec.append(dict(runtime_path=r['runtime_path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
    cmd=['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for row in d['sources']:
        text=row.pop('content',None)
        if text is not None:
            p=OUT/'common-metadata-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
            assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk identity only; source review ranges separate. No serving framework imports or device actions.')
    (OUT/'common-metadata-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),container_id=d['container_id'],image=d['image'])))
if __name__=='__main__':main()
