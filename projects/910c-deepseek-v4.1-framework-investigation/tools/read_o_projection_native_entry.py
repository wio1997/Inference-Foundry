#!/usr/bin/env python3
"""Read-only recheck of actual O projection and SP/mHC consumer entry."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE

OUT=Path('audit/execution-search')
def main():
    prior=json.loads((OUT/'attention-output-source-verification.json').read_text())['files']
    prior+=json.loads((OUT/'metadata-readers-execution-identity.json').read_text())['sources']
    prior+=json.loads(Path('/Users/wio/work/branch-fusion-audit-20261104/full-review/flow-index.json').read_text())['files']
    names=['attention/context_parallel/dsa_v41_cp.py','attention/context_parallel/dsa_cp.py','attention/dsa_v1.py','models/layer/attention/layer.py','ops/cv_linear.py','models/deepseek_v41/model.py','models/common/ops/sequence_parallel.py']
    spec=[]
    for name in names:
        row=next(r for r in prior if r['runtime_path'].endswith('/'+name));p=Path(row['local_path']);spec.append(dict(runtime_path=row['runtime_path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for row in d['sources']:
        body=row.pop('content',None)
        if body is not None:
            p=OUT/'o-projection-entry-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
    d.update(scope='Current disk identities only, explicit OA/OB route ranges separate. No framework import or nativebit admission.',new_NPU_work=0,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest());(OUT/'o-projection-native-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matched=sum(r.get('matches_local',False) for r in d['sources']),new_NPU_work=0)))
if __name__=='__main__':main()
