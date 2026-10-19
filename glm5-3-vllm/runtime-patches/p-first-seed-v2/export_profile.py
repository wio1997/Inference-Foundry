#!/usr/bin/env python3
"""Snapshot stopped P rank0 profile, CPU-export, retain exact trace artifacts."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
    root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32')
    gate=json.loads((root/'profile-request-v1/profile-request.json').read_text());assert gate['completed'] and gate['profile_stopped']
    selected=list((root/'profile').glob('dp0_pp0_tp0_dcp0_ep0_rank0_*_ascend_pt'));assert len(selected)==1
    source=selected[0];snapshot=root/'profile-rank0-raw-v1'/source.name;assert not snapshot.exists();assert shutil.disk_usage(root).free>5*1024**3
    files=sorted(p for p in source.rglob('*') if p.is_file());assert files and not any(p.is_symlink() for p in files)
    snapshot.mkdir(parents=True);manifest=[]
    for p in files:
        digest=sha(p);q=snapshot/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(q)==digest==sha(p)
        manifest.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest})
    out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
    (out/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    cmd=['docker','run','--rm','--pull','never','--network','none','--name','glm53-seed-p-rank0-export-v1','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','64g','--pids-limit','4096','--tmpfs','/tmp:rw,size=2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','--device','/dev/davinci_manager','-v',str(snapshot)+':/profiles/'+source.name,'-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/timeout',image,'--signal=TERM','--kill-after=15','420','python3','-c',"from torch_npu.profiler.profiler import analyse; analyse('/profiles/"+source.name+"', max_process_number=4, export_type='text')"]
    start=time.monotonic()
    with (out/'export-stdout.txt').open('xb') as stdout,(out/'export-stderr.txt').open('xb') as stderr:result=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=450)
    report={'rc':result.returncode,'duration_s':time.monotonic()-start,'argv':cmd,'selected_rank':0,'compute_devices':[],'raw_files':len(manifest),'raw_bytes':sum(p['bytes'] for p in manifest),'source_snapshot':str(snapshot),'scope':'P rank0 request path evidence only; no all-rank new-copy claim or performance timing.'}
    targets=list(snapshot.rglob('trace_view.json'))+list(snapshot.rglob('communication_matrix.json'))+list(snapshot.rglob('kernel_details.csv'))+list(snapshot.rglob('api_statistic.csv'))
    copied=[]
    for p in targets:
        q=out/p.relative_to(snapshot);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);copied.append({'path':str(q.relative_to(out)),'bytes':q.stat().st_size,'sha256':sha(q)})
    report['exported_files']=copied
    (out/'export-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
    assert result.returncode==0 and any(p['path'].endswith('trace_view.json') for p in copied)


if __name__=='__main__':main()
