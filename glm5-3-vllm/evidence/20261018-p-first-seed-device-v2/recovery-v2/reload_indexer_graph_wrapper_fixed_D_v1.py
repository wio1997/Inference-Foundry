from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys

r=Path(__file__).resolve().parent
repo=r.parents[3]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m=module('transport',r/'run_prefix_count_service_v1.py')
g=module('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
ssh=['ssh','-S',m.control,'-oBatchMode=yes','-p','60011','root@61.241.77.34']
build=json.loads((r/'indexer-k-graph-overlap-build-manifest-v3.json').read_text())
check=json.loads((r/'indexer-k-graph-overlap-build-check-v3.json').read_text())
assert build['identity']==check['build'] and check['previous_live_build_invalid_inactive']
assert json.loads((r/'indexer-k-graph-overlap-cpu-check-v2.json').read_text())['passed']==16
inactive=json.loads(next((r/'indexer-wrapper-inactive-v1').glob('*/metrics.json')).read_text())
assert len(inactive['all16'])==16 and all(z['selected']==z['max_entries']==0 for z in inactive['all16'])
x=m.collect(r/'snapshot_indexer_graph_inactive_D_v1.py','indexer_graph_inactive_D_snapshot_v1','indexer-graph-inactive-D-snapshot-v1',timeout=120)
assert x['completed']
run=next((r/'indexer-graph-inactive-D-snapshot-v1').glob('*/result.json')).parent
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',m.control,'--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-indexer-K-graph-inactive-D'],check=True)
receipt=json.loads((run/'artifacts-backup.json').read_text())
assert hashlib.sha256(Path(receipt['local_archive']).read_bytes()).hexdigest()==receipt['sha256']
stage=m.collect(Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-indexer-K-graph-build-v3/stage_overlay.py'),'indexer_k_graph_stage_v3','indexer-k-graph-stage-v3',timeout=90)
assert stage['root']==build['remote_root']
plan=json.loads((r/'indexer-k-graph-D-plan-v1.json').read_text())
res=plan['reservation']
pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
uid,allowed,expiry=g.fresh_lease(pod,res,datetime.datetime.now(datetime.timezone.utc))
assert uid==res['expected_uid'] and not pod['metadata'].get('ownerReferences')
pods=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items']
assert [p['metadata']['uid'] for p in pods if p['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in p['spec']['containers'])]==[uid]
x=m.collect(r/'stop_indexer_graph_inactive_D_v1.py','indexer_graph_inactive_D_stop_v1','indexer-graph-inactive-D-stop-v1',timeout=110)
assert x['empty_physical_cards']==8
plan['runtime_patches']=build['patches']
plan['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live2-20261019/node33'
plan['container'].update(name='glm53-indexer-k-graph-33-live2-20261019',lifetime_seconds=min(7200,allowed-60))
assert plan['container']['lifetime_seconds']>=3600
argv=plan['serve_argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-indexer-k-graph-d-live2-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,
            objective='Correct exact IndexerWrapper.impl ownership boundary for MTP K-only graph; native P/proxy retained',
            notes=['Previous build had zero eligible/selected calls on all16, so functional numeric only native fallback; no quality/formal executed.',
                   'Wrapper source pin e14080c6 and model-facing delegation CPU protocol added. New immutable build; no running-source edit.',
                   'Only snapshot-checked idle exact D replaced. Current P70e58 and stock proxy, original parameters/precision/operators/SDMA retained.'])
target=r/'indexer-k-graph-D-plan-v2.json'
assert not target.exists()
target.write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(target),'--control',m.control,'--output',str(r/'indexer-k-graph-D-launch-v2')],check=True)
print(json.dumps(dict(D_reloaded=True,P_and_proxy_retained=True,build=build['identity'],holder_UID=uid,model_lifetime_s=plan['container']['lifetime_seconds'])),flush=True)
