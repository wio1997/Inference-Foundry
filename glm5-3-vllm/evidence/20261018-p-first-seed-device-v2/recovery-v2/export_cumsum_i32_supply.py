"""Preserve stopped D raw data, CPU-export all ranks, correlate target Cumsum and supply diagnosis."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import shutil
import socket
import subprocess
import tarfile
import time

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-cumsum-i32-live1-20261019/node33')
gate=json.loads((root/'cumsum-i32-supply-profile-v1/result.json').read_text())
assert gate['completed'] and gate['profile_stopped'] and gate['ranks']==list(range(16))
assert all(x['complete_answer_pass'] for x in gate['rows'])
dirs=[Path(p) for p in gate['new_profile_dirs']]
files=sorted(p for d in dirs for p in d.rglob('*') if p.is_file())
assert all(p.parent==root/'profile' and not p.is_symlink() for p in dirs)
assert all(not p.is_symlink() for p in files)
assert sum(p.stat().st_size for p in files)<4*1024**3 and shutil.disk_usage(root).free>8*1024**3
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
snapshot=root/'cumsum-i32-supply-rank0-raw-v1';snapshot.mkdir(exist_ok=False)
manifest=[]
for p in files:
    digest=sha(p)
    if '_rank0_' in p.relative_to(root/'profile').parts[0]:
        q=snapshot/p.relative_to(root/'profile');q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q);assert sha(q)==digest
    manifest.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest})
(out/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copyfile(root/'cumsum-i32-supply-profile-v1/result.json',out/'profile-result.json')
with tarfile.open(out/'all16-raw.tar.gz','w:gz',compresslevel=3) as tar:
    for p,record in zip(files,manifest):
        tar.add(p,arcname=record['path'],recursive=False)
        assert sha(p)==record['sha256']
image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
argv=['docker','run','--rm','--pull','never','--network','none','--name','glm53-cumsum-i32-supply-rank0-export-v1',
      '--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','16','--memory','64g','--pids-limit','4096',
      '--tmpfs','/tmp:rw,size=2g','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1',
      '--device','/dev/davinci_manager','-v',str(snapshot)+':/profiles',
      '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','/usr/bin/timeout',image,
      '--signal=TERM','--kill-after=15','300','python3','-c',
      "from torch_npu.profiler.profiler import analyse; analyse('/profiles',max_process_number=1,export_type='text')"]
start=time.monotonic()
with (out/'export-stdout.txt').open('xb') as a,(out/'export-stderr.txt').open('xb') as b:
    proc=subprocess.run(argv,stdout=a,stderr=b,timeout=330)
assert proc.returncode==0,proc.returncode

directory=next(snapshot.glob('*_rank0_*_ascend_pt'))
trace=next(directory.rglob('trace_view.json'))
rank_out=out/'rank0';rank_out.mkdir()
shutil.copyfile(trace,rank_out/'trace_view.json')
for name in ['kernel_details.csv','api_statistic.csv','communication_matrix.json']:
    for p in directory.rglob(name):shutil.copyfile(p,rank_out/name)
report={'completed':True,'export_rc':proc.returncode,'export_elapsed_s':time.monotonic()-start,'argv':argv,
    'raw_files':len(manifest),'raw_bytes':sum(x['bytes'] for x in manifest),'raw_ranks':gate['ranks'],
    'raw_archive_sha256':sha(out/'all16-raw.tar.gz'),'raw_archive_bytes':(out/'all16-raw.tar.gz').stat().st_size,
    'trace_sha256':sha(trace),'trace_bytes':trace.stat().st_size,
    'compute_devices_used':[],'exported_ranks':[0],
    'scope':'Stopped all16 raw preserved with per-file SHA; CPU-only rank0 export for candidate-int32-prefix target Cumsum and supply diagnosis. No service performance estimate and no additional GPU requests.'}
(out/'export-result.json').write_text(json.dumps(report,indent=2)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
