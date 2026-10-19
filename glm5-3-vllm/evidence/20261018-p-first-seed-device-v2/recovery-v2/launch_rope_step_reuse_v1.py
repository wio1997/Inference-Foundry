from pathlib import Path
import datetime, hashlib, importlib.util, json, subprocess, sys
r=Path(__file__).resolve().parent;repo=r.parents[3]
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('transport',r/'run_prefix_count_service_v1.py')
g=module('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
ssh=['ssh','-S',m.control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
check=json.loads((r/'rope-step-reuse-build-check-v1.json').read_text());assert check['CPU_checks']==13
build=json.loads((r/'rope-step-reuse-build-manifest-v1.json').read_text());assert build['identity']==check['build']
for node in [33,32]:
 p=next((r/('early-sample-publication-v4-snapshot-'+str(node)+'-v1')).glob('*/artifacts-backup.json'));v=json.loads(p.read_text());assert hashlib.sha256(Path(v['local_archive']).read_bytes()).hexdigest()==v['sha256']
assert json.loads((r/'early-sample-publication-reviewed-verdict-v2.json').read_text())['decision']=='REJECT_DEFAULT_OFF_EARLY_PUBLICATION'
x=m.collect(Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-rope-step-reuse-build-v1/stage_overlay.py'),'rope_step_reuse_stage_v1','rope-step-reuse-stage-v1',timeout=90);assert x['root']==build['remote_root']
plan=json.loads((r/'early-sample-publication-D-plan-v2.json').read_text());res=plan['reservation']
pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
uid,allowed,expiry=g.fresh_lease(pod,res,datetime.datetime.now(datetime.timezone.utc));assert uid==res['expected_uid'] and not pod['metadata'].get('ownerReferences')
pods=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items']
assert [p['metadata']['uid'] for p in pods if p['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in p['spec']['containers'])]==[uid]
proxy=m.collect(r/'check_publication_P_proxy_v4.py','rope_reuse_retained_P_proxy_v1','rope-step-reuse-retained-P-proxy-v1',timeout=90,node=32);assert proxy['ready']
x=m.collect(r/'stop_publication_D_for_rope_reuse_v1.py','publication_D_stop_for_rope_reuse_v1','publication-D-stop-for-rope-reuse-v1',timeout=120);assert x['empty_physical_cards']==8
plan['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33'
plan['container'].update(name='glm53-rope-step-reuse-33-live1-20261019',lifetime_seconds=min(7200,allowed-60));assert plan['container']['lifetime_seconds']>=3600
plan['env'].update(GLM53_EARLY_SAMPLE_PUBLICATION='0',GLM53_EARLY_SAMPLE_COPY_SHADOW='0',GLM53_ROPE_STEP_REUSE_PROBE='1')
plan['runtime_patches']=build['patches'];argv=plan['serve_argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-rope-step-reuse-d-live1-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,objective='Finite same-step SFA RoPE reuse by cache-only Indexer, original native publication/retirement',notes=['Retained loaded P84bd and stock proxy unchanged. Only exact snapshotted idle owned D replaced.','Same pinned model, precision, parameters, operators and SDMA. No cross-step memo; original independent Indexer buffers retained.'])
p=r/'rope-step-reuse-D-plan-v1.json';assert not p.exists();p.write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(p),'--control',m.control,'--output',str(r/'rope-step-reuse-D-launch-v1')],check=True)
print(json.dumps(dict(D_launched=True,P_retained=True,build=build['identity'],lifetime_s=plan['container']['lifetime_seconds'])),flush=True)
