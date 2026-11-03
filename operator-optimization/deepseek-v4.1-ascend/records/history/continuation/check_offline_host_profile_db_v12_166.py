from pathlib import Path
import json, os, socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
JOB=P/'analysis/offline-db-v12'
launch=json.loads((P/'analysis/offline-db-v12-launch.json').read_text())
pid=launch['host_pid']
stat=Path('/proc',str(pid),'stat').read_text()
fields=stat[stat.rfind(')')+2:].split()
proc=dict(pid=pid,state=fields[0],starttime_ticks=fields[19],starttime_matches=fields[19]==launch['host_starttime_ticks'],cmdline=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()[:400])
status=json.loads((JOB/'status.json').read_text()) if (JOB/'status.json').exists() else None
out=P/'analysis/offline-db-v12.stdout'; err=P/'analysis/offline-db-v12.stderr'
res=dict(launch=launch,process=proc,job_exists=JOB.exists(),job_entries=sorted(x.name for x in JOB.iterdir())[:30] if JOB.exists() else [],
         status=status,stdout_bytes=out.stat().st_size if out.exists() else None,stderr_bytes=err.stat().st_size if err.exists() else None,
         stdout_tail=out.read_text(errors='replace')[-1200:] if out.exists() else None,
         stderr_tail=err.read_text(errors='replace')[-1200:] if err.exists() else None)
print(json.dumps(res,ensure_ascii=False,indent=2))
