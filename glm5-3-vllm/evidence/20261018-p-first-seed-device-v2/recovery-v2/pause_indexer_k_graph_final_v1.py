"""Snapshot, SHA-backup, stop exact owned CIDs and end UID-bound reservations."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,subprocess,sys,time
r=Path(__file__).resolve().parent;repo=r.parents[3]
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ssh=['ssh','-S',m.control,'-oBatchMode=yes','-oStrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
assert (r/'indexer-k-graph-reviewed-verdict-v1.json').exists() or (r/'indexer-k-graph-closed-without-formal-v1.json').exists()
leases=[]
for role,node,v in [('D',33,2),('P',32,1)]:
 plan=json.loads((r/('indexer-k-graph-'+role+'-plan-v'+str(v)+'.json')).read_text());res=plan['reservation']
 pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
 assert pod['metadata']['uid']==res['expected_uid'] and pod['metadata']['labels']['owner']==res['owner_label']
 assert not pod['metadata'].get('ownerReferences') and pod['spec']['nodeName']==res['node_name']
 leases.append(dict(node=node,reservation=res,initial_phase=pod['status']['phase']))
receipts=[]
for node in [33,32]:
 out='indexer-k-graph-final-snapshot-v1-'+str(node)
 x=m.collect(r/'snapshot_indexer_k_graph_final_v1.py','indexer_k_graph_final_snapshot_v1_'+str(node),out,timeout=120,node=node)
 assert x['completed'] and x['node']==node
 run=next((r/out).glob('*/result.json')).parent
 subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',m.control,'--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-indexer-K-graph-final','--chunk-size-mib','8'],check=True)
 receipt=json.loads((run/'artifacts-backup.json').read_text())
 assert Path(receipt['local_archive']).stat().st_size==receipt['bytes'] and hashlib.sha256(Path(receipt['local_archive']).read_bytes()).hexdigest()==receipt['sha256']
 receipts.append(dict(node=node,archive=receipt))
 print(json.dumps(dict(phase='final_snapshot_backed_up',node=node,bytes=receipt['bytes'],sha256=receipt['sha256'])),flush=True)
stops=[]
for node in [33,32]:
 x=m.collect(r/'stop_indexer_k_graph_final_v1.py','indexer_k_graph_final_stop_v1_'+str(node),'indexer-k-graph-final-stop-v1-'+str(node),timeout=150,node=node)
 assert x['completed'] and all(not z['after']['Running'] for z in x['rows']);stops.append(x)
 print(json.dumps(dict(phase='exact_owned_CIDs_stopped',node=node)),flush=True)
for lease in leases:
 res=lease['reservation']
 pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=30))
 assert pod['metadata']['uid']==res['expected_uid'] and pod['metadata']['labels']['owner']==res['owner_label'] and not pod['metadata'].get('ownerReferences')
 if pod['status']['phase'] not in ['Failed','Succeeded']:
  patch=[dict(op='test',path='/metadata/uid',value=res['expected_uid']),dict(op='test',path='/metadata/resourceVersion',value=pod['metadata']['resourceVersion']),dict(op='replace',path='/spec/activeDeadlineSeconds',value=1)]
  import shlex
  subprocess.run(ssh+['kubectl patch pod -n default '+res['name']+' --type=json -p '+shlex.quote(json.dumps(patch))],check=True,capture_output=True,timeout=30)
 lease['deadline_seconds']=1
for lease in leases:
 res=lease['reservation'];end=time.monotonic()+60
 while True:
  pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'],timeout=15))
  assert pod['metadata']['uid']==res['expected_uid']
  if pod['status']['phase'] in ['Failed','Succeeded']:break
  assert time.monotonic()<end
  time.sleep(2)
 lease['final_phase']=pod['status']['phase'];lease['reason']=pod['status'].get('reason')
 print(json.dumps(dict(phase='own_holder_released',node=lease['node'],UID=res['expected_uid'],final_phase=lease['final_phase'])),flush=True)
report=dict(completed=True,paused_at_user_request=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshots=receipts,stops=stops,leases=leases,scope='Current experiment closed. All exact own model/proxy containers stopped and own UID-bound finite holders ended after SHA backups. Access pods and system/other workloads untouched. No follow-up automation or new experiment.')
(r/'indexer-k-graph-pause-receipt-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(completed=True,paused_at_user_request=True)),flush=True)
