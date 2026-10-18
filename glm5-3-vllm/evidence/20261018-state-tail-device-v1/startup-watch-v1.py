from pathlib import Path
import concurrent.futures,datetime,json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-state-tail-device-v1'
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
def inspect(n):
 plan=json.loads((e/f'launch-plans/{"P" if n==32 else "D"}.json').read_text())
 code='''import datetime,json,pathlib,subprocess,urllib.request
root=pathlib.Path(RUN)
p=json.loads((root/'plan.json').read_text());l=json.loads((root/'launch.json').read_text());cid=l['container_id']
d=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=12))[0]
assert d['Id']==cid and d['Name']=='/'+p['container']['name'] and d['Image']==p['image']['id']
r={'node':NODE,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cid':cid,'state':d['State'],'ready':False}
logs=subprocess.run(['docker','logs','--timestamps','--tail','100',cid],capture_output=True,text=True,timeout=15)
lines=(logs.stdout+'\\n'+logs.stderr).splitlines();r['tail']=lines[-12:];r['errors']=[s for s in lines if any(x in s for x in ['Traceback','AttributeError','ValueError','RuntimeError','ERROR'])][-12:]
try:
 with urllib.request.urlopen('http://172.16.12.NODE:PORT/v1/models',timeout=2) as f:models=json.load(f)
 r['ready']=any(x['id']=='glm5' for x in models['data']);r['models']=models
except Exception as ex:r['api_error']=str(ex)
print(json.dumps(r))
'''.replace('RUN',repr(plan['remote']['run_dir'])).replace('NODE',str(n)).replace('PORT',str(plan['container']['port']))
 r=subprocess.run(ssh+[f'kubectl exec -i -n default {pods[n]} -- nsenter -t 1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=40)
 assert r.returncode==0,r.stderr
 return json.loads(r.stdout)
started=min(datetime.datetime.fromisoformat(json.loads((e/f'{r}-launch/launch.json').read_text())['started_host_utc']) for r in ('P','D'))
previous=None;last=0
with (e/'startup-watch.jsonl').open('x') as out:
 while True:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(inspect,(32,33)))
  elapsed=(datetime.datetime.now(datetime.timezone.utc)-started).total_seconds()
  value={'elapsed_since_launch_s':elapsed,'nodes':rows};out.write(json.dumps(value)+'\n');out.flush()
  signature=tuple((r['ready'],r['state']['Status'],tuple(r['errors'])) for r in rows)
  if signature!=previous or time.monotonic()-last>=55:
   print(json.dumps(value),flush=True);previous=signature;last=time.monotonic()
  if all(r['ready'] for r in rows):status='both_ready';break
  if any(not r['state']['Running'] for r in rows):status='startup_instance_failed';break
  if elapsed>=900:status='readiness_watchdog_expired';break
  time.sleep(15)
 value['status']=status;(e/'startup-outcome.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'outcome':status,'elapsed_s':elapsed}),flush=True)
