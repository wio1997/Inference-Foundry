"""Read-only preflight for deciding whether an isolated real-model diagnostic can coexist."""
import datetime
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
remote = '''import json,subprocess,datetime
target='wio-vllm-v41-repro169-20261028'
stage='wio-vllm-v41-repro169-stage-20261028'
def call(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=20)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1500:]}
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device_work':False}
for name in [target,stage]:
 p=call(['docker','inspect',name])
 if p['returncode']==0:
  j=json.loads(p['stdout'])[0]
  out[name]={'id':j['Id'],'running':j['State']['Running'],'image':j['Image']}
 else:out[name]=p
out['npu_status']=call(['docker','exec',stage,'npu-smi','info'])
out['owned_service_processes']=call(['docker','top',target,'-eo','pid,comm'])
health="""import json,urllib.request
o=urllib.request.build_opener(urllib.request.ProxyHandler({}));out={}
for path in ['health','metrics']:
 try:
  with o.open('http://127.0.0.1:8172/'+path,timeout=4) as r:
   out[path+'_http']=r.status
   if path=='metrics':
    out['selected_metrics']=[l for l in r.read().decode().splitlines() if not l.startswith('#') and any(k in l for k in ['num_requests_running','num_requests_waiting','gpu_cache_usage_perc','kv_cache_usage_perc'])]
 except Exception as e:out[path+'_error']=type(e).__name__+': '+str(e)
print(json.dumps(out))
"""
out['health_metrics']=call(['docker','exec',stage,'python3','-c',health])
print(json.dumps(out))
'''
p=subprocess.run([SSH,shlex.join(['python3','-c',remote])],capture_output=True,text=True,timeout=100)
if p.returncode:
    raise RuntimeError(p.stderr[-1500:])
j=json.loads(p.stdout)
(ROOT/'diagnostic-20261029/whole-model-preflight.json').write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(j,ensure_ascii=False,indent=2))
