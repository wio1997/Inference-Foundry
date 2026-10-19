"""Resume read-only readiness checks after a transport timeout; keep exact CIDs."""
from pathlib import Path
import json,subprocess,time
r=Path(__file__).resolve().parent
ssh=['ssh','-S','/tmp/glm53-active-20261019.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
wait_launch=time.monotonic()
while not list((r/'rope-step-reuse-P-launch-v1').rglob('launch.json')):
 assert time.monotonic()-wait_launch<300;time.sleep(2)
new_launch=json.loads(next((r/'rope-step-reuse-P-launch-v1').rglob('launch.json')).read_text())
expected={32:new_launch['container_id'],33:'163c7646c30ab819514f112e0af510934aa592305f3ec37bd7e3b77a3edc2bdc'}
remote='''import datetime,json,pathlib,subprocess,urllib.request
root=pathlib.Path(c['root']);cid=c['CID'];plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text())
d=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=10))[0]
assert d['Id']==launch['container_id']==cid and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(launch['started_host_utc'])).total_seconds()
row=dict(node=c['node'],CID=cid,running=d['State']['Running'],ready=False,elapsed=elapsed,remaining_model_s=plan['container']['lifetime_seconds']-elapsed)
try:
 with urllib.request.urlopen('http://172.16.12.%d:%d/v1/models'%(c['node'],29481 if c['node']==32 else 29482),timeout=2) as f:data=json.load(f)
 row['ready']=any(z['id']=='glm5' for z in data['data'])
except Exception as exc:row['api_error']=str(exc)
if not row['ready']:
 q=subprocess.run(['docker','logs','--timestamps','--tail','60',cid],capture_output=True,text=True,timeout=10)
 lines=(q.stdout+'\\n'+q.stderr).splitlines();row['tail']=lines[-4:];row['errors']=[z for z in lines if any(v in z for v in ['Traceback','AssertionError','RuntimeError','AttributeError','ERROR'])][-6:]
 if c['node']==32 and row['running'] and elapsed>=1200:
  q=subprocess.run(['docker','stop','-t','30',cid],capture_output=True,text=True,timeout=40)
  assert q.returncode==0;row['stopped_exact_own_failed_start']=True;row['running']=False
print(json.dumps(row))
'''
states={};errors=[];start=time.monotonic()
with (r/'rope-step-reuse-P-startup-watch-v1.jsonl').open('x') as out:
    while len(states)<2:
        assert time.monotonic()-start<900
        for node in [32,33]:
            if node in states:continue
            c=dict(node=node,CID=expected[node],root=('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-p-live1-20261019/node32' if node==32 else '/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33'))
            code='c='+repr(c)+'\n'+remote
            try:
                q=subprocess.run(ssh+['kubectl --request-timeout=20s exec -i -n default '+pods[node]+' -- nsenter -t 1 -m -- python3 -'],input=code,capture_output=True,text=True,timeout=60)
                assert q.returncode==0,q.stderr
                row=json.loads(q.stdout)
            except (subprocess.TimeoutExpired,AssertionError,json.JSONDecodeError) as exc:
                row=dict(node=node,CID=expected[node],transport_or_probe_error=str(exc));errors.append(row)
            out.write(json.dumps(row)+'\n');out.flush();print(json.dumps(row),flush=True)
            if 'ready' in row and (row['ready'] or not row['running']):states[node]=row
        if len(states)<2:time.sleep(20)
report=dict(both_ready=all(z['ready'] for z in states.values()),nodes=states,
            recovery='Fresh same-config P-only reload after original finite timeout; D retained. Transport probe failures recorded; D never stopped by readiness watchdog.',probe_errors=errors)
target=r/'rope-step-reuse-P-startup-outcome-v1.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
assert report['both_ready'],report
