"""Replace only naturally expired own P; retain the loaded verified D."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,subprocess,sys
r=Path(__file__).resolve().parent;repo=r.parents[3]
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('transport',r/'run_prefix_count_service_v1.py');g=module('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
ssh=['ssh','-S',m.control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
assert json.loads((r/'rope-step-reuse-cached-download-recovery-v1.json').read_text())['original_requests_repeated']==0
assert json.loads((r/'rope-step-reuse-correctness-outcome-v1.json').read_text())['cached16']
for node in [32,33]:
 label='rope-step-reuse-before-P-reload-snapshot-'+str(node)+'-v1'
 x=m.collect(r/'snapshot_rope_step_reuse_before_P_reload_v1.py','rope_step_reuse_before_P_reload_snapshot_v1',label,timeout=120,node=node);assert x['node']==node
 run=next((r/label).glob('*/result.json')).parent
 subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',m.control,'--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-rope-step-reuse-before-P-reload'],check=True)
 receipt=json.loads((run/'artifacts-backup.json').read_text());assert hashlib.sha256(Path(receipt['local_archive']).read_bytes()).hexdigest()==receipt['sha256']
plan=json.loads((r/'early-sample-publication-P-plan-v1.json').read_text());res=plan['reservation']
pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30));uid,allowed,expiry=g.fresh_lease(pod,res,datetime.datetime.now(datetime.timezone.utc))
assert uid==res['expected_uid'] and not pod['metadata'].get('ownerReferences')
pods=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items']
assert [p['metadata']['uid'] for p in pods if p['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in p['spec']['containers'])]==[uid]
x=m.collect(r/'check_expired_P_for_reuse_v1.py','expired_P_for_rope_reuse_v1','rope-step-reuse-expired-P-v1',timeout=100,node=32);assert x['empty_physical_cards']==8
plan['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-p-live1-20261019/node32'
plan['container'].update(name='glm53-rope-step-reuse-32-live1-20261019',lifetime_seconds=min(7200,allowed-60))
assert plan['container']['lifetime_seconds']>=1800,'Insufficient budget for measured ~650s startup and finite controls/formal/backup; refresh owned holder before launch'
plan['env']['GLM53_ROPE_STEP_REUSE_PROBE']='0'
argv=plan['serve_argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-rope-step-reuse-p-live1-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,objective='Native same-config P replacement after original finite watchdog; retain loaded D for metadata reuse research',notes=['Only original own P and proxy naturally expired with exit124, OOMFalse; snapshots SHA verified before replacement. No job stopped.','Same immutable P overlays/config, W8A8C8/int8 KV/Indexer, SDMA, PIECEWISE[256,1024], sharedFalse and K1. Existing owned finite holder reused.','New P requires fresh PD smoke before quality controls/formal. Old completed runs retained; no NPU rerun to recover interrupted cached download.'])
p=r/'rope-step-reuse-P-plan-v1.json';assert not p.exists();p.write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(p),'--control',m.control,'--output',str(r/'rope-step-reuse-P-launch-v1')],check=True)
print(json.dumps(dict(P_reloaded=True,D_retained=True,holder_UID=uid,lifetime_s=plan['container']['lifetime_seconds'])),flush=True)
