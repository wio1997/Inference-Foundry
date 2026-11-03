#!/usr/bin/env python3
import sys,shlex,subprocess,json,datetime
from pathlib import Path
root=Path(__file__).resolve().parent
host,script,label=sys.argv[1:4]
connector=Path('/Users/wio/work/166-vllm-migration-20261030/ssh166.sh') if host=='166' else Path('/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh')
code=Path(script).read_text()
p=subprocess.run([str(connector),'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=180)
r={'host':host,'script':script,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(root/(label+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'host':host,'returncode':p.returncode,'receipt':str(root/(label+'.json')),'stdout_tail':p.stdout[-600:] if p.returncode else 'saved to receipt','stderr_tail':p.stderr[-2500:]},ensure_ascii=False),flush=True)
sys.exit(p.returncode)
