import json,sys,time
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102');from control import invoke
H=Path(__file__).resolve().parent
code=r'''import json,pathlib,subprocess,urllib.request,time,collections
p=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168')
h=json.loads((p/'SUBMITTED.json').read_text());q=pathlib.Path('/proc/'+str(h['pid'])+'/stat');live=q.exists() and q.read_text().rsplit(')',1)[1].split()[19]==h['start_ticks'];out={'observed_unix':time.time(),'handle':h,'handle_live':live,'status':json.loads((p/'STATUS.json').read_text()) if (p/'STATUS.json').exists() else None}
a=json.loads(subprocess.check_output(['docker','inspect','wio-vllm-v41-repro171-20261102']))[0];out['main']={k:a['State'].get(k) for k in ['Running','StartedAt','Status','Pid']}
try:out['health_http']=urllib.request.urlopen('http://127.0.0.1:30075/health',timeout=3).status
except Exception as e:out['health_error']=str(e)
for name in ['controller.log','client.stderr']:
 q=p/name
 if q.exists():
  with q.open('rb') as f:f.seek(max(0,q.stat().st_size-5000));out[name]=f.read().decode(errors='replace')
q=p.parent.parent/'logs/service.log'
if q.exists():
 with q.open('rb') as f:f.seek(max(0,q.stat().st_size-7000));out['service_log_tail']=f.read().decode(errors='replace')
out['metadata_worker_dirs']=len(list((p/'workers').glob('*')));out['history_finals']=len(list((p/'workers').glob('*/FINAL.json')))
print(json.dumps(out))'''
r=invoke('target',code,timeout=35);(H/'observation-latest.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(H/('observation-'+str(int(time.time()))+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
small={k:v for k,v in r.items() if k not in ['handle','service_log_tail']};small['service_log_tail']=r.get('service_log_tail','')[-4500:];print(json.dumps(small,ensure_ascii=False))
