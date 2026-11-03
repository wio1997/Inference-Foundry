from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030')
EXPECTED='hostprof-20261030T212812Z-360e0c72'
latest=json.loads((ROOT/'host-profile-latest.json').read_text());assert latest['run_id']==EXPECTED
p=Path('/proc',str(latest['controller_pid']),'stat')
live=False;observed=None
try:
 raw=p.read_text();f=raw[raw.rfind(')')+2:].split();observed=dict(state=f[0],starttime_ticks=f[19]);live=f[0]!='Z' and f[19]==latest['controller_starttime_ticks']
except FileNotFoundError:pass
root=Path(latest['run_root']);small={}
for name in ['controller-state.json','controller-finished.json','controller-error.json','top-level-error.json','diagnostic-identity.json','restore-result.json','trace-inventory.json','capture-index.json']:
 p=root/name
 if p.exists():
  x=json.loads(p.read_text());small[name]=({k:v for k,v in x.items() if k not in ['files','profiler_info_fields','ownership']} if name in ['trace-inventory.json','capture-index.json','controller-state.json'] else x)
for phase in ['warmup','baseline','profile']:
 p=root/'runtime'/('client-'+phase)/'window-status.json'
 if p.exists():small['client-'+phase]={k:v for k,v in json.loads(p.read_text()).items() if k not in ['counter_deltas_by_engine']}
containers=[]
for name in ['wio-vllm-v41-repro166-20261030','wio-vllm-v41-'+EXPECTED]:
 q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
 if q.returncode==0:
  d=json.loads(q.stdout)[0];containers.append(dict(name=name,id=d['Id'],running=d['State']['Running'],status=d['State']['Status'],exit_code=d['State']['ExitCode']))
log=root/'runtime/logs/service.log'
service_tail=[]
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-5000));service_tail=f.read().decode(errors='replace').splitlines()[-22:]
err=Path(latest['stderr']);error_tail=err.read_text()[-5000:] if err.exists() else ''
result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),launch=latest,controller_live_verified_by_pid_start=live,controller_proc=observed,states=small,containers=containers,service_tail=service_tail,controller_stderr_tail=error_tail)
print(json.dumps(result,ensure_ascii=False))
