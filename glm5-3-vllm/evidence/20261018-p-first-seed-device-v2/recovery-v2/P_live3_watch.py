from pathlib import Path
import json,subprocess,time
r=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=r/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';p=json.loads((e/'P-live3-plan.json').read_text());ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34'];code="""import datetime,json,pathlib,subprocess,urllib.request
root=pathlib.Path(RUN);p=json.loads((root/'plan.json').read_text());l=json.loads((root/'launch.json').read_text());cid=l['container_id'];d=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=12))[0];assert d['Id']==cid and d['Name']=='/'+p['container']['name'] and d['Image']==p['image']['id']
r={'node_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_host_s':(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(l['started_host_utc'])).total_seconds(),'cid':cid,'state':d['State'],'ready':False};q=subprocess.run(['docker','logs','--timestamps','--tail','100',cid],text=True,capture_output=True,timeout=15);lines=(q.stdout+'\\n'+q.stderr).splitlines();r['tail']=lines[-6:];r['errors']=[s for s in lines if any(x in s for x in ['Traceback','AttributeError','ValueError','RuntimeError','ERROR'])][-8:]
try:
 with urllib.request.urlopen('http://172.16.12.32:29481/v1/models',timeout=2) as f:models=json.load(f)
 r['ready']=any(x['id']=='glm5' for x in models['data'])
except Exception as ex:r['api_error']=str(ex)
print(json.dumps(r))
""".replace('RUN',repr(p['remote']['run_dir']))
with (e/'P-live3-startup-watch.jsonl').open('x') as out:
 while True:
  q=subprocess.run(ssh+['kubectl exec -i -n default sglang-node-access-32 -- nsenter -t 1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=40);assert q.returncode==0,q.stderr;row=json.loads(q.stdout);out.write(json.dumps(row)+'\n');out.flush();print(json.dumps({'elapsed':row['elapsed_host_s'],'ready':row['ready'],'running':row['state']['Running'],'errors':row['errors'],'tail':row['tail'][-2:]}),flush=True)
  if row['ready'] or not row['state']['Running'] or row['elapsed_host_s']>=900:break
  time.sleep(20)
(e/'P-live3-startup-outcome.json').write_text(json.dumps(row,indent=2)+'\n');assert row['ready'],row
