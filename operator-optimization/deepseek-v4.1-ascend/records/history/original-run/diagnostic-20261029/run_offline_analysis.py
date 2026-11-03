"""CPU-only parse of this recapture, executed inside the existing own stage."""
from pathlib import Path
import json
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
STAGE='wio-vllm-v41-repro169-stage-20261028'
REMOTE='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'
scripts={name:(HERE/name).read_text() for name in ('summarize_metadata.py','summarize_profile_scopes.py')}
code=r'''
from pathlib import Path
import json,subprocess,sys,os
r=Path(REMOTE);runtime=r/'runtime';assert runtime.is_dir()
temporary=r/'cpu-analysis-tmp';temporary.mkdir(exist_ok=True)
os.environ['TMPDIR']=str(temporary)
progress=json.loads((runtime/'profile-client-results/progress.json').read_text())
assert progress['state']=='REQUESTS_COMPLETE_EXPORT_PENDING' and progress['profile_stop_http']==200
for name,source in SCRIPTS.items():
    p=r/name
    if p.exists():assert p.read_text()==source,('Do not overwrite independent source',name)
    else:p.write_text(source)
import torch
torch.set_num_threads(2)
from torch_npu.profiler.profiler import analyse
print('OFFLINE_PROFILE_PARSE_START',flush=True)
analyse(str(runtime/'diagnostic-traces'),max_process_number=2)
print('OFFLINE_PROFILE_PARSE_RETURNED',flush=True)
commands=[('metadata-analysis.stdout',[sys.executable,str(r/'summarize_metadata.py'),'--root',str(runtime)]),('scope-summary.json',[sys.executable,str(r/'summarize_profile_scopes.py'),str(runtime)])]
for name,args in commands:
    proc=subprocess.run(args,capture_output=True,text=True,timeout=900)
    (r/name).write_text(proc.stdout)
    (r/(name+'.stderr')).write_text(proc.stderr)
    if proc.returncode:raise RuntimeError((args,proc.stderr[-3000:]))
print(json.dumps({'offline_parse_completed':True,'kernel_details_count':len(list((runtime/'diagnostic-traces').glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))),'scope_summary':str(r/'scope-summary.json'),'metadata_summary':str(runtime/'metadata-summary.json')}),flush=True)
'''
prefix='REMOTE='+repr(REMOTE)+'\nSCRIPTS='+repr(scripts)+'\n'
remote='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec',STAGE,'python3','-u','-c',prefix+code])+'))'
proc=subprocess.run([SSH,shlex.join(['python3','-c',remote])],capture_output=True,text=True,timeout=2400)
(HERE/'offline-analysis.stdout').write_text(proc.stdout)
(HERE/'offline-analysis.stderr').write_text(proc.stderr)
print(proc.stdout[-2500:])
if proc.returncode:print(proc.stderr[-4000:]);raise SystemExit(proc.returncode)
