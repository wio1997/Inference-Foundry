"""CPU-only, bounded streaming scope attribution over completed recapture."""
from pathlib import Path
import json
import shlex
import subprocess
HERE=Path(__file__).resolve().parent
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
STAGE='wio-vllm-v41-repro169-stage-20261028'
REMOTE='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'
source=(HERE/'summarize_profile_scopes_v2.py').read_text()
compile(source,'summarize_profile_scopes_v2.py','exec')
code=r'''
from pathlib import Path
import subprocess,sys,json
r=Path(REMOTE);p=r/'summarize_profile_scopes_v2.py'
if p.exists():assert p.read_text()==SOURCE
else:p.write_text(SOURCE)
out=r/'scope-summary-v2.json';err=r/'scope-summary-v2.stderr'
assert not out.exists(),'Preserve existing scope result'
with out.open('x') as target,err.open('x') as errors:
    result=subprocess.run([sys.executable,str(p),str(r/'runtime')],stdout=target,stderr=errors,timeout=1600)
if result.returncode:raise RuntimeError(err.read_text()[-2500:])
d=json.loads(out.read_text());ranks=d.get('ranks',[])
print(json.dumps({'status':'SCOPE_SUMMARY_WRITTEN','ranks':len(ranks),'trace_statuses':{x:sum(k.get('trace_status')==x for k in ranks) for x in sorted({k.get('trace_status') for k in ranks})},'output':str(out)}))
'''
prefix='REMOTE='+repr(REMOTE)+'\nSOURCE='+repr(source)+'\n'
outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec',STAGE,'python3','-u','-c',prefix+code])+'))'
r=subprocess.run([SSH,shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=1800)
(HERE/'scope-v2.stdout').write_text(r.stdout);(HERE/'scope-v2.stderr').write_text(r.stderr)
print(r.stdout[-2000:])
if r.returncode:print(r.stderr[-3000:]);raise SystemExit(r.returncode)
