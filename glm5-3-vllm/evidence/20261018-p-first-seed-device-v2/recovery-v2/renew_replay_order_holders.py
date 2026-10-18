from pathlib import Path
import copy,datetime,json,subprocess,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34'];pods={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'};owner='codex-glm53-replay-order-live2-20261018';receipts=[]
for node,role in [(32,'P'),(33,'D')]:
 assert (r/('sfa-replay-order-live1-'+role+'-backup.json')).exists();finished=json.loads(next((r/('sfa-replay-order-live1-'+role+'-finish')).glob('*/metrics.json')).read_text());assert all(not x['after']['Running'] for x in finished['stopped'])
 old=json.loads((r/('sfa-replay-order-lease'+str(node)+'.json')).read_text());name=old['metadata']['name'];uid=old['metadata']['uid'];current=subprocess.run(ssh+['kubectl get pod -n default '+name+' --ignore-not-found -o json'],capture_output=True,text=True,timeout=30);assert current.returncode==0,current.stderr
 assigned=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+old['spec']['nodeName']+' -o json'],timeout=30))['items'];assert all(not sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in x['spec'].get('containers',[])) or x['metadata']['uid']==uid or x['status']['phase'] in ['Failed','Succeeded'] or (x['metadata'].get('labels',{}).get('owner')==owner) for x in assigned)
 smi=subprocess.check_output(ssh+['kubectl exec -i -n default '+pods[node]+' -- nsenter -t 1 -m -- npu-smi info'],timeout=30).decode();assert smi.count('No running processes found in NPU')==8;(r/('sfa-replay-order-live2-idle-'+str(node)+'.txt')).write_text(smi)
 if current.stdout.strip():
  obj=json.loads(current.stdout);assert obj['metadata']['uid']==uid and obj['metadata']['labels']['owner']==old['metadata']['labels']['owner'] and not obj['metadata'].get('ownerReferences')
  eviction={'apiVersion':'policy/v1','kind':'Eviction','metadata':{'name':name,'namespace':'default'},'deleteOptions':{'preconditions':{'uid':uid},'gracePeriodSeconds':10}}
  q=subprocess.run(ssh+['kubectl create --raw /api/v1/namespaces/default/pods/'+name+'/eviction -f -'],input=json.dumps(eviction),capture_output=True,text=True,timeout=30);assert q.returncode==0,q.stderr
  subprocess.run(ssh+['kubectl wait --for=delete pod/'+name+' -n default --timeout=60s'],check=True,capture_output=True,text=True,timeout=75);deletion={'old_name':name,'old_UID':uid,'eviction_return':q.stdout,'UID_precondition':True}
 else:deletion={'old_name':name,'old_UID':uid,'already_absent':True}
 fresh='glm53-replay-order-'+str(node)+'-live2-20261018';container={k:copy.deepcopy(old['spec']['containers'][0][k]) for k in ['name','image','command','args','resources','securityContext'] if k in old['spec']['containers'][0]};container['imagePullPolicy']='Never'
 spec={'nodeName':old['spec']['nodeName'],'restartPolicy':'Never','activeDeadlineSeconds':7200,'terminationGracePeriodSeconds':10,'automountServiceAccountToken':False,'containers':[container]}
 for k in ['tolerations','securityContext','schedulerName']:
  if k in old['spec']:spec[k]=copy.deepcopy(old['spec'][k])
 doc={'apiVersion':'v1','kind':'Pod','metadata':{'name':fresh,'namespace':'default','labels':{'app':'glm53-replay-order','owner':owner}},'spec':spec};(r/('sfa-replay-order-live2-reservation-'+str(node)+'.json')).write_text(json.dumps(doc,indent=2)+'\n')
 q=subprocess.run(ssh+['kubectl get pod -n default '+fresh+' --ignore-not-found -o json'],capture_output=True,text=True,timeout=30);assert q.returncode==0
 if not q.stdout.strip():
  q=subprocess.run(ssh+['kubectl create -f - -o json'],input=json.dumps(doc),capture_output=True,text=True,timeout=40);assert q.returncode==0,q.stderr
 subprocess.run(ssh+['kubectl wait --for=condition=Ready pod/'+fresh+' -n default --timeout=90s'],capture_output=True,text=True,check=True,timeout=105)
 pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+fresh+' -o json'],timeout=30));assert pod['metadata']['labels']['owner']==owner and pod['spec']['nodeName']==spec['nodeName'] and pod['status']['phase']=='Running' and pod['spec']['activeDeadlineSeconds']==7200 and not pod['metadata'].get('ownerReferences')
 (r/('sfa-replay-order-live2-lease'+str(node)+'.json')).write_text(json.dumps(pod,indent=2)+'\n');deletion.update(new_name=fresh,new_UID=pod['metadata']['uid'],finite_seconds=7200);receipts.append(deletion);print(json.dumps(deletion),flush=True)
(r/'sfa-replay-order-live2-holder-receipt.json').write_text(json.dumps({'nodes':receipts,'scope':'Only two exact owned oldholders evicted withUID precondition afterSHAbackup/modelstop/NPUidle; finite2h newholders, nootherjobs oraccesspods stopped.'},indent=2)+'\n')
