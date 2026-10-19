from pathlib import Path
import concurrent.futures,datetime,json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan'); e=root/'glm5-3-vllm/evidence/20261018-state-tail-device-v1'
control='/tmp/glm53-active-20261018.sock'
ssh=['ssh','-S',control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
def command(cmd,**kw):return subprocess.run(ssh+[cmd],capture_output=True,check=True,timeout=45,**kw)
def stage(n):
 c=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node',str(n),'--script','/Users/wio/work/glm53-execution-evidence-backup/20261018-state-tail-probe-build-v5/stage_overlay.py','--label','tail_live1_stage','--control',control,'--output',str(e/f'stage{n}'),'--timeout','90']
 r=subprocess.run(c,capture_output=True,text=True,cwd=root,timeout=140);(e/f'stage{n}.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 return {'node':n,'staged':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:print(json.dumps({'staging':list(pool.map(stage,(32,33)))}),flush=True)
docs=[]
for n in (32,33):
 p=json.loads((e/f'review-plans/{"P" if n==32 else "D"}.json').read_text())
 listed=json.loads(command(f'kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-{n} -o json').stdout)
 for row in listed['items']:
  count=sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in row['spec'].get('containers',[]))
  assert not count or row['status']['phase'] in ('Failed','Succeeded'),'Other live NPU claim; no preemption'
 smi=command(f'kubectl exec -i -n default {pods[n]} -- nsenter -t 1 -m -- npu-smi info').stdout.decode()
 assert smi.count('No running processes found in NPU')==8,'Device process present; no start'
 doc=json.loads((root/f'glm5-3-vllm/tools/node{n}-reservation.json').read_text())
 doc['metadata'].update(name=p['reservation']['name'],labels={'app':'glm53-state-tail','owner':p['reservation']['owner_label']})
 doc['metadata']['annotations']['purpose']='Owned sustained GLM53 state-tail research window; bounded7200s, no model in holder'
 doc['spec']['activeDeadlineSeconds']=7200;doc['spec']['containers'][0]['command']=['/bin/sleep','7200'];docs.append(doc)
(e/'reservation-intent.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_documents':docs,'no_other_live_claims_or_device_processes':True},indent=2)+'\n')
r=command('kubectl create -f -',input=json.dumps({'apiVersion':'v1','kind':'List','items':docs}).encode());(e/'reservation-create.txt').write_bytes(r.stdout+r.stderr)
end=time.monotonic()+120
while True:
 ready=[]
 for n,doc in zip((32,33),docs):
  pod=json.loads(command('kubectl get pod -n default '+doc['metadata']['name']+' -o json').stdout)
  (e/f'lease{n}.json').write_text(json.dumps(pod,indent=2)+'\n')
  ready.append({'node':n,'uid':pod['metadata']['uid'],'phase':pod['status']['phase'],'ascend_real':pod['metadata'].get('annotations',{}).get('huawei.com/AscendReal')})
 if all(p['phase']=='Running' and p['ascend_real'] for p in ready):break
 assert time.monotonic()<end,ready
 time.sleep(3)
(e/'reservation-receipt.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'after':ready,'deadline_seconds':7200,'models_loaded':False},indent=2)+'\n');print(json.dumps({'reservations':ready}),flush=True)
subprocess.run(['python3',str(root/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py'),'--build-manifest','/Users/wio/work/glm53-execution-evidence-backup/20261018-state-tail-probe-build-v5/manifest.json','--run-id','tail-c1-live1-20261018','--lease32',str(e/'lease32.json'),'--lease33',str(e/'lease33.json'),'--output',str(e/'launch-plans')],cwd=root,check=True)
def launch(role):
 c=['python3',str(root/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(e/f'launch-plans/{role}.json'),'--control',control,'--output',str(e/f'{role}-launch')]
 r=subprocess.run(c,capture_output=True,text=True,cwd=root,timeout=240);(e/f'{role}-launch-collector.txt').write_text(r.stdout+r.stderr)
 return {'role':role,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(launch,('P','D')))
(e/'launch-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True)
assert all(r['rc']==0 for r in results)
