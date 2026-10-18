from pathlib import Path
import copy,hashlib,json,subprocess
r=Path(__file__).resolve().parent
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
old=json.loads((r/'moe-target-live1-lease32.json').read_text());name=old['metadata']['name'];uid=old['metadata']['uid'];assert uid=='a41b366f-4dcd-41d5-aaaf-25ce3bbaa25e'
b=json.loads(next((r/'cumsum-i32-P-refresh-snapshot').glob('*/artifacts-backup.json')).read_text());a=Path(b['local_archive']);assert a.stat().st_size==b['bytes'] and hashlib.sha256(a.read_bytes()).hexdigest()==b['sha256']
assigned=json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+old['spec']['nodeName']+' -o json'],timeout=25))['items'];assert all(not sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in z['spec'].get('containers',[])) or z['metadata']['uid']==uid or z['status']['phase'] in ['Failed','Succeeded'] for z in assigned)
current=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+name+' -o json'],timeout=20));assert current['metadata']['uid']==uid and current['metadata']['labels']['owner']==old['metadata']['labels']['owner'] and not current['metadata'].get('ownerReferences')
code="""import datetime,json,subprocess,urllib.request
for node,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,port),timeout=5) as f:s=f.read().decode()
 rows=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(rows)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in rows)
items=[('6bb5441031253d21c5fe2c0a9d9ef2e6f9bfc96b55682f92b4d82e8624292123','glm53-moe-target-32-live1-20261019-official-proxy'),('13c520313d3369bafc2db811dca35bc93afdfc2ae7ed3572fc1518366c38aceb','glm53-moe-target-32-live1-20261019')]
rows=[]
for cid,name in items:
 d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
 if cid.startswith('6bb544'):assert not d['HostConfig'].get('Devices') and d['Name'].startswith('/glm53-')
 else:assert d['Name']=='/'+name
 if d['State']['Running']:subprocess.run(['docker','stop','-t','30',cid],check=True,timeout=45,capture_output=True)
 after=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert not after['State']['Running'];rows.append({'CID':cid,'name':after['Name'],'state':after['State']})
s=subprocess.check_output(['npu-smi','info'],text=True,timeout=20);assert s.count('No running processes found in NPU')==8
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stopped':rows,'idle_cards':8,'D_untouched':True}))
"""
q=subprocess.run(ssh+['kubectl exec -i -n default sglang-node-access-32 -- nsenter -t 1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=110);assert q.returncode==0,q.stderr;stop=json.loads(q.stdout);(r/'cumsum-i32-P-stop-for-refresh.json').write_text(json.dumps(stop,indent=2)+'\n')
ev={'apiVersion':'policy/v1','kind':'Eviction','metadata':{'name':name,'namespace':'default'},'deleteOptions':{'preconditions':{'uid':uid},'gracePeriodSeconds':10}}
q=subprocess.run(ssh+['kubectl create --raw /api/v1/namespaces/default/pods/'+name+'/eviction -f -'],input=json.dumps(ev),text=True,capture_output=True,timeout=25);assert q.returncode==0,q.stderr
subprocess.run(ssh+['kubectl wait --for=delete pod/'+name+' -n default --timeout=60s'],check=True,capture_output=True,timeout=75)
fresh='glm53-cumsum-i32-32-live2-20261019';owner='codex-glm53-cumsum-i32-P-live2-20261019'
container={k:copy.deepcopy(old['spec']['containers'][0][k]) for k in ['name','image','command','args','resources','securityContext'] if k in old['spec']['containers'][0]};container['imagePullPolicy']='Never'
spec={'nodeName':old['spec']['nodeName'],'restartPolicy':'Never','activeDeadlineSeconds':7200,'terminationGracePeriodSeconds':10,'automountServiceAccountToken':False,'containers':[container]}
for k in ['tolerations','securityContext','schedulerName']:
 if k in old['spec']:spec[k]=copy.deepcopy(old['spec'][k])
doc={'apiVersion':'v1','kind':'Pod','metadata':{'name':fresh,'namespace':'default','labels':{'app':'glm53-cumsum-i32','owner':owner}},'spec':spec};assert not subprocess.check_output(ssh+['kubectl get pod -n default '+fresh+' --ignore-not-found -o json'],timeout=20).strip()
q=subprocess.run(ssh+['kubectl create -f - -o json'],input=json.dumps(doc),text=True,capture_output=True,timeout=30);assert q.returncode==0,q.stderr
subprocess.run(ssh+['kubectl wait --for=condition=Ready pod/'+fresh+' -n default --timeout=60s'],check=True,capture_output=True,timeout=75)
pod=json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+fresh+' -o json'],timeout=20));assert pod['metadata']['labels']['owner']==owner and not pod['metadata'].get('ownerReferences')
(r/'cumsum-i32-P-live2-lease32.json').write_text(json.dumps(pod,indent=2)+'\n');receipt={'old_name':name,'old_UID':uid,'new_name':fresh,'new_UID':pod['metadata']['uid'],'D_CID_unchanged':'416f0dc10e34a115e2537cb62e46e2eb98d30825642ab4ba36d098c072cf11f5','finite_seconds':7200,'backup_sha256':b['sha256'],'scope':'Exact own P and no-NPU stock proxy stopped idle after SHA snapshot; UID-conditioned holder renewal on32 only. D remains loaded.'};(r/'cumsum-i32-P-holder-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
