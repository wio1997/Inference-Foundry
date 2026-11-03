from pathlib import Path
from datetime import datetime,timezone
import json,socket,subprocess
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
launch=json.loads((P/'analysis/offline-db-v12-launch.json').read_text());p=Path('/proc',str(launch['host_pid']),'stat');live=False;proc=None
try:
 raw=p.read_text();f=raw[raw.rfind(')')+2:].split();proc=dict(state=f[0],starttime_ticks=f[19]);live=f[0]!='Z' and f[19]==launch['host_starttime_ticks']
except FileNotFoundError:pass
job=Path(launch['job_root']);status=json.loads((job/'status.json').read_text()) if (job/'status.json').exists() else None
index=json.loads((job/'database-index.json').read_text()) if (job/'database-index.json').exists() else None
summary=[]
if index:
 for d in index['databases']:
  summary.append(dict(path=d['path'],bytes=d['bytes'],sha256=d['sha256'],counts={k:v['count'] for k,v in d['wanted'].items()}))
result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),launch=launch,host_process_live_verified_by_pid_start=live,host_proc=proc,
 status={k:v for k,v in status.items() if k!='databases'} if status else None,
 database_summary=summary,source_raw_unchanged=index.get('original_raw_unchanged') if index else None,
 stdout_tail=Path(launch['stdout']).read_text()[-5000:] if Path(launch['stdout']).exists() else '',stderr_tail=Path(launch['stderr']).read_text()[-3000:] if Path(launch['stderr']).exists() else '')
print(json.dumps(result,ensure_ascii=False))
