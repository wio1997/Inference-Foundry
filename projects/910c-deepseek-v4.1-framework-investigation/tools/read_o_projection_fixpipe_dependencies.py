#!/usr/bin/env python3
"""Recheck current SDK L0C->L1 rounding/layout entry without compiling a kernel."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE

OUT=Path('audit/execution-search')
if __name__=='__main__':
    base='/usr/local/Ascend/cann-9.2.0/aarch64-linux/asc/'
    names=[base+n for n in ['include/interface/kernel_operator_fixpipe_intf.h','impl/basic_api/kernel_operator_fixpipe_intf_impl.h','impl/basic_api/dav_c220/kernel_operator_fixpipe_v2_impl.h']]
    spec=[]
    for name in names:
        p=OUT/'dfc-gemm-extra-runtime-source'/name.lstrip('/');spec.append(dict(runtime_path=name,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
    if r.returncode:raise RuntimeError(r.stderr)
    d=json.loads(r.stdout)
    for row in d['sources']:
        b=row.pop('content',None)
        if b is not None:
            p=OUT/'o-projection-fixpipe-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(b);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
    d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='InstalledSDK declaration/implementation identity only; devicearch applicability/actual original rounding/nativebit separate.');(OUT/'o-projection-fixpipe-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matched=sum(r.get('matches_local',False) for r in d['sources']))))
