from pathlib import Path
import concurrent.futures,datetime,importlib.util,json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2';old=root/'glm5-3-vllm/evidence/20261018-p-first-adopt-device-v1';control='/tmp/glm53-active-20261018.sock'
ssh=['ssh','-S',control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}
def command(cmd,**kw):return subprocess.run(ssh+[cmd],capture_output=True,check=True,timeout=50,**kw)
def collect(n,script,label,out,args=(),timeout=150):
 c=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node',str(n),'--script',str(script),'--label',label,'--control',control,'--output',str(out),'--timeout',str(timeout)]
 for a in args:c+=['--script-arg='+a]
 r=subprocess.run(c,capture_output=True,text=True,cwd=root,timeout=timeout+80);(e/(label+'.txt')).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 return json.loads(next(out.rglob('metrics.json')).read_text())
cpu=json.loads(next((e/'cpu-check-v1').rglob('metrics.json')).read_text());assert cpu['rc']==0,cpu['stderr']
for n in (32,33):assert json.loads(next((e/f'stage{n}').rglob('metrics.json')).read_text())['staged'] is True
oldplans={n:json.loads((old/f'launch-plans/{"P" if n==32 else "D"}.json').read_text()) for n in (32,33)}
# Ownership inventory is fresh; only exact old UID may be live.
for n,p in oldplans.items():
 listed=json.loads(command(f'kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-{n} -o json').stdout)
 (e/f'pre-reload-pods{n}.json').write_text(json.dumps(listed,indent=2)+'\n')
 for row in listed['items']:
  count=sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in row['spec'].get('containers',[]))
  assert not count or row['status']['phase'] in ('Failed','Succeeded') or (row['metadata']['uid']==p['reservation']['expected_uid'] and row['metadata']['labels'].get('owner')==p['reservation']['owner_label'] and not row['metadata'].get('ownerReferences')),'Other live claim; no preemption'
def finish(n):
 p=oldplans[n]
 return collect(n,root/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/finish.py',f'seed_old{n}_replaced',e/f'old{n}-replaced',['--run-dir',p['remote']['run_dir'],'--reason','stage-complete'])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:finished=list(pool.map(finish,(32,33)))
(e/'old-stop-summary.json').write_text(json.dumps(finished,indent=2)+'\n');print(json.dumps({'old_exact_cids_stopped':[{k:r[k] for k in ('container_id','before_state','after_state')} for r in finished]}),flush=True)
# No reservation release until both precise model instances are stopped and cards idle.
for n,p in oldplans.items():
 smi=command(f'kubectl exec -i -n default {pods[n]} -- nsenter -t 1 -m -- npu-smi info').stdout.decode();(e/f'idle-before-reload{n}.txt').write_text(smi);assert smi.count('No running processes found in NPU')==8,'Other device process present; no release/start'
 q=subprocess.run(ssh+['kubectl get pod -n default '+p['reservation']['name']+' -o json'],capture_output=True,timeout=30)
 if q.returncode:
  assert b'NotFound' in q.stderr;continue
 pod=json.loads(q.stdout);res=p['reservation'];assert pod['metadata']['uid']==res['expected_uid'] and pod['metadata']['labels']['owner']==res['owner_label'] and not pod['metadata'].get('ownerReferences')
 options={'apiVersion':'v1','kind':'DeleteOptions','preconditions':{'uid':res['expected_uid']},'gracePeriodSeconds':0}
 r=command('kubectl delete --raw /api/v1/namespaces/default/pods/'+res['name']+' -f -',input=json.dumps(options).encode());(e/f'old{n}-holder-delete.json').write_bytes(r.stdout)
docs=[];run_id='seed-live1-20261018'
for n in (32,33):
 listed=json.loads(command(f'kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-{n} -o json').stdout)
 assert all(not sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in row['spec'].get('containers',[])) or row['status']['phase'] in ('Failed','Succeeded') for row in listed['items'])
 doc=json.loads((root/f'glm5-3-vllm/tools/node{n}-reservation.json').read_text());doc['metadata'].update(name=f'glm53-p-first-seed-{n}-{run_id}',labels={'app':'glm53-p-first-seed','owner':'codex-p-first-'+run_id});doc['metadata']['annotations']['purpose']='Owned finite3h research reservation; model<=2h; precise cleanup and renewable holder deadline'
 doc['spec']['activeDeadlineSeconds']=10800;doc['spec']['containers'][0]['command']=['/bin/sleep','infinity'];docs.append(doc)
(e/'reservation-intent.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_documents':docs,'no_other_live_claims_or_device_processes':True},indent=2)+'\n')
r=command('kubectl create -f -',input=json.dumps({'apiVersion':'v1','kind':'List','items':docs}).encode());(e/'reservation-create.txt').write_bytes(r.stdout+r.stderr)
end=time.monotonic()+120
while True:
 leases={}
 for n,doc in zip((32,33),docs):
  p=json.loads(command('kubectl get pod -n default '+doc['metadata']['name']+' -o json').stdout);leases[n]=p;(e/f'lease{n}.json').write_text(json.dumps(p,indent=2)+'\n')
 if all(p['status']['phase']=='Running' and p['metadata'].get('annotations',{}).get('huawei.com/AscendReal') for p in leases.values()):break
 assert time.monotonic()<end;time.sleep(3)
spec=importlib.util.spec_from_file_location('fresh',root/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
packet=json.loads((root/'glm5-3-vllm/diagnostics/p-first-state-v1/review-plan.json').read_text());manifest=json.loads((e/'build-manifest.json').read_text());(e/'launch-plans').mkdir()
for role,n in [('P',32),('D',33)]:
 p=json.loads((root/packet['base_configs'][role]['path']).read_text());res=p['reservation'];res.update(name=leases[n]['metadata']['name'],owner_label=leases[n]['metadata']['labels']['owner']);uid,lifetime,expiry=m.fresh_lease(leases[n],res,datetime.datetime.now(datetime.timezone.utc));res['expected_uid']=uid
 p.update(state_tail_probe=True,schema='glm53-capacity-launch/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,objective='Per-request confirmed first-token history and unverified P MTP1 proposal passed to native D rejection verification',notes=['Existing sustained execution authorization,32/33 only.','P K1 and D K5; original precision/model/operators.','Native control/adopt/seed on same loaded instances; no benefit claimed before complete measurement.','Holder3h, model2h, readiness15min; no preemption.'])
 p['remote']['run_dir']=f'/data/wio/glm5-3-vllm-execution/20261018-state-tail/{run_id}/node{n}';p['container'].update(name=f'glm53-p-first-seed-{n}-{run_id}',lifetime_seconds=lifetime)
 p['env'].update(GLM53_STATE_TAIL_PROBE='0',GLM53_P_FIRST_ADOPT='1' if role=='D' else '0',GLM53_P_FIRST_ADOPT_TRACE='1' if role=='D' else '0',GLM53_P_FIRST_DRAFT_EXPORT='1' if role=='P' else '0')
 p['runtime_patches']=manifest['patches'];i=p['serve_argv'].index('--kv-transfer-config')+1;kv=json.loads(p['serve_argv'][i]);kv['engine_id']=f'glm53-p-first-seed-{role.lower()}-{run_id}';p['serve_argv'][i]=json.dumps(kv,separators=(',',':'));(e/f'launch-plans/{role}.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'fresh_holders':{n:p['metadata']['uid'] for n,p in leases.items()},'starting_native_models':True}),flush=True)
def launch(role):
 r=subprocess.run(['python3',str(root/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(e/f'launch-plans/{role}.json'),'--control',control,'--output',str(e/f'{role}-launch')],cwd=root,capture_output=True,text=True,timeout=240);(e/f'{role}-launch-collector.txt').write_text(r.stdout+r.stderr);return {'role':role,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(launch,('P','D')))
(e/'launch-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True);assert all(r['rc']==0 for r in results)
