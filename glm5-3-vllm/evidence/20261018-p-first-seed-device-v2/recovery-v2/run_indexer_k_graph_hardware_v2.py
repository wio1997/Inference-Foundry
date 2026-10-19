"""Actual all16 transport proof after finite quality review, before formal timing."""
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
review=json.loads((r/'indexer-k-graph-quality-reviewed-v2.json').read_text())
assert review['proceed_finite_hardware_and_formal_review']
paths=json.loads(next((r/'indexer-k-graph-paths-full-v2').glob('*/metrics.json')).read_text())
assert paths['all16'] and all(z['N']==[6,12,18,24] for z in paths['rows'])
leases=[]
for role,node,version in [('P',32,1),('D',33,2)]:
    plan=json.loads((r/('indexer-k-graph-'+role+'-plan-v'+str(version)+'.json')).read_text())
    res=plan['reservation']
    pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
    uid,allowed,expiry=g.fresh_lease(pod,res,datetime.datetime.now(datetime.timezone.utc))
    assert uid==res['expected_uid'] and not pod['metadata'].get('ownerReferences')
    pods=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'],timeout=30))['items']
    active=[p['metadata']['uid'] for p in pods if p['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in p['spec']['containers'])]
    assert active==[uid]
    pre=m.collect(r/'preflight_indexer_k_graph_v2.py','indexer_k_graph_hardware_preflight_v2_'+str(node),'indexer-k-graph-hardware-preflight-v2-'+str(node),timeout=100,node=node)
    assert pre['remaining_s']>2400
    leases.append(dict(node=node,holder_UID=uid,allowed_s=allowed,model_remaining_s=pre['remaining_s']))
(r/'indexer-k-graph-hardware-ownership-v2.json').write_text(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),leases=leases),indent=2)+'\n')
profile=m.collect(r/'profile_indexer_k_graph_copies_all16_v2.py','indexer_k_graph_copy_profile_all16_v2','indexer-k-graph-copy-profile-all16-v2',timeout=800)
assert profile['completed'] and len(profile['profiles'])==2
print(json.dumps(dict(phase='all16_C1_C4_copy_profile_stopped',rows=len(profile['rows']),raw_bytes=sum(z['raw_bytes'] for z in profile['profiles']))),flush=True)
export=m.collect(r/'export_indexer_k_graph_copy_profiles_v2.py','indexer_k_graph_copy_export_all16_v2','indexer-k-graph-copy-export-all16-v2',timeout=1300)
assert export['all16_C1_C4_new_copies_SDMA']
run=next((r/'indexer-k-graph-copy-export-all16-v2').glob('*/result.json')).parent
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',m.control,
                '--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-indexer-K-graph-all16-copy','--chunk-size-mib','8'],check=True)
receipt=json.loads((run/'artifacts-backup.json').read_text())
assert hashlib.sha256(Path(receipt['local_archive']).read_bytes()).hexdigest()==receipt['sha256']
report=dict(reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,
            all16_C1_C4_new_copies_SDMA=True,total_copies=sum(z['added_copies'] for z in export['rows']),rows=export['rows'],
            archive=receipt,P_CID=review['P_CID'],D_CID=review['D_CID'],
            scope='Stopped actual C1/C4 diagnostic profiles, all16 strict SDMA_SQE new input-copy chains, raw and full trace archives SHA-verified. Native transport config retained. Not service performance and not every-future-byte proof. Primary formal review and fresh idle/ownership check still required.')
(r/'indexer-k-graph-hardware-reviewed-v2.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows','archive']}),flush=True)
