from pathlib import Path
import datetime,importlib.util,json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');old=root/'glm5-3-vllm/evidence/20261018-state-tail-device-v1';e=root/'glm5-3-vllm/evidence/20261018-p-first-adopt-device-v1';control='/tmp/glm53-active-20261018.sock'
end=time.monotonic()+180
while not list((e/'cpu-check-v5').glob('*/metrics.json')):
 assert time.monotonic()<end;time.sleep(2)
cpu=json.loads(next((e/'cpu-check-v5').glob('*/metrics.json')).read_text());assert cpu['rc']==0,cpu['stderr']
assert json.loads(next((e/'stage33').glob('*/metrics.json')).read_text())['staged'] is True
ssh=['ssh','-S',control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
plan=json.loads((old/'launch-plans/D.json').read_text());res=plan['reservation']
listing=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-33 -o json'],timeout=30))
for p in listing['items']:
 count=sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in p['spec'].get('containers',[]))
 assert not count or p['metadata']['uid']==res['expected_uid'] or p['status']['phase'] in ('Failed','Succeeded'),'Other live claim; no action'
pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30));(e/'lease33-reused.json').write_text(json.dumps(pod,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('fresh',root/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
uid,lifetime,expiry=m.fresh_lease(pod,res,datetime.datetime.now(datetime.timezone.utc));assert uid==res['expected_uid']
manifest=json.loads((e/'build-manifest.json').read_text())
plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),objective='Per-request initial P-token history adoption; same-instance native/control comparison',holder_expiry_utc=expiry,lifetime_refreshed_client_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
plan['notes']=['Existing sustained execution authorization; reuse exact owned reservation and P model.','Opt-in per request; unsupported/max1 native fallback. No early publication.','Keep loaded for continued correctness/work-count/end-to-end testing; finite watchdog.']
plan['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/adopt-live1-20261018/node33'
plan['container'].update(name='glm53-adopt-first-33-live1-20261018',lifetime_seconds=lifetime)
plan['env'].update(GLM53_STATE_TAIL_PROBE='0',GLM53_P_FIRST_ADOPT='1',GLM53_P_FIRST_ADOPT_TRACE='1')
plan['runtime_patches']=manifest['patches'];idx=plan['serve_argv'].index('--kv-transfer-config')+1;kv=json.loads(plan['serve_argv'][idx]);kv['engine_id']='glm53-adopt-first-d-live1-20261018';plan['serve_argv'][idx]=json.dumps(kv,separators=(',',':'))
(e/'launch-plans').mkdir();(e/'launch-plans/D.json').write_text(json.dumps(plan,indent=2)+'\n');(e/'launch-plans/P.json').write_bytes((old/'launch-plans/P.json').read_bytes());(e/'P-launch').mkdir();(e/'P-launch/launch.json').write_bytes((old/'P-launch/launch.json').read_bytes())
(e/'P-reused.json').write_text(json.dumps({'source_launch':str(old/'P-launch'),'container_id':json.loads((old/'P-launch/launch.json').read_text())['container_id'],'restarted':False},indent=2)+'\n')
c=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node','33','--script',str(root/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/finish.py'),'--label','tail_live1_D_replaced','--control',control,'--output',str(old/'D-replaced'),'--timeout','150','--script-arg=--run-dir','--script-arg=/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node33','--script-arg=--reason','--script-arg=stage-complete']
r=subprocess.run(c,cwd=root,capture_output=True,text=True,timeout=200);(old/'D-replaced-collector.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
raw=subprocess.check_output(ssh+['kubectl exec -i -n default sglang-tuning-access-33-20261013 -- nsenter -t 1 -m -- npu-smi info'],timeout=30).decode();(e/'D-idle-before-replacement.txt').write_text(raw);assert raw.count('No running processes found in NPU')==8
# Refresh remaining lifetime after the precise stop, preserving cleanup margin.
pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30));uid,lifetime,expiry=m.fresh_lease(pod,res,datetime.datetime.now(datetime.timezone.utc));assert uid==res['expected_uid'];plan['container']['lifetime_seconds']=lifetime;plan['holder_expiry_utc']=expiry;(e/'launch-plans/D.json').write_text(json.dumps(plan,indent=2)+'\n')
r=subprocess.run(['python3',str(root/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(e/'launch-plans/D.json'),'--control',control,'--output',str(e/'D-launch')],cwd=root,capture_output=True,text=True,timeout=240);(e/'D-launch-collector.txt').write_text(r.stdout+r.stderr);print(json.dumps({'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'P_reused':True}),flush=True);assert r.returncode==0
