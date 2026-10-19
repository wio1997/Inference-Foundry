"""Preserve the loaded P/proxy; snapshot and replace only exact own D."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,subprocess,sys
r=Path(__file__).resolve().parent;repo=r.parents[3];control='/tmp/glm53-active-20261019.sock'
ssh=['ssh','-S',control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
controls=json.loads(next((r/'early-sample-publication-native-controls-before-v4-v1').glob('*/metrics.json')).read_text());assert controls['completed']
check=json.loads((r/'early-sample-publication-build-check-v4.json').read_text());assert check['passed']==23
build=json.loads((r/'early-sample-publication-build-manifest-v4.json').read_text());assert build['identity']==check['build']
assert json.loads(next((r/'early-sample-publication-stage-v4-33').glob('*/metrics.json')).read_text())['root']==build['remote_root']
for node in [33,32]:
    x=m.collect(r/'snapshot_early_publication_v3.py','early_publication_v3_snapshot_v1','early-sample-publication-v3-snapshot-'+str(node)+'-v1',timeout=120,node=node)
    assert x['node']==node
    run=next((r/('early-sample-publication-v3-snapshot-'+str(node)+'-v1')).glob('*/result.json')).parent
    subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',control,
                    '--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-early-publication-v3-final-snapshot'],check=True)
    receipt=json.loads((run/'artifacts-backup.json').read_text());assert hashlib.sha256(Path(receipt['local_archive']).read_bytes()).hexdigest()==receipt['sha256']
plan=json.loads((r/'early-sample-publication-D-plan-v1.json').read_text());res=plan['reservation']
pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
uid,allowed,expiry=g.fresh_lease(pod,res,datetime.datetime.now(datetime.timezone.utc));assert uid==res['expected_uid'] and not pod['metadata'].get('ownerReferences')
pods=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items']
active=[z['metadata']['uid'] for z in pods if z['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in z['spec']['containers'])]
assert active==[uid];lifetime=min(7200,allowed-60);assert lifetime>=3600
x=m.collect(r/'stop_publication_D_for_v4.py','early_publication_D_stop_for_v4','early-sample-publication-D-stop-for-v4',timeout=120,node=33)
assert x['empty_physical_cards']==8
plan['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live2-20261019/node33'
plan['container'].update(name='glm53-early-publication-33-live2-20261019',lifetime_seconds=lifetime)
plan['runtime_patches']=build['patches'];argv=plan['serve_argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-early-publication-d-live2-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,
            objective='Early publication with independent original physical2-batch admission before input reuse; retain loaded P/proxy',
            notes=['P84bd and stock proxy stay loaded and unmodified. Same pinned model, precision, existing operators and SDMA.','v3 quality24/26 full64 equal; Native controls cover quality1 but not quality0. No v3 formal performance and no causal attribution to admission.','v4 has23 CPU checks, explicit oldest-native-late admission gate and retained GPU-step cap2. All other5 overlays unchanged fromv3.'])
target=r/'early-sample-publication-D-plan-v2.json';assert not target.exists();target.write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(target),'--control',control,
                '--output',str(r/'early-sample-publication-D-launch-v2')],check=True)
print(json.dumps(dict(D_reloaded=True,P_retained=True,holder_uid=uid,lifetime_s=lifetime,build=build['identity'])),flush=True)
