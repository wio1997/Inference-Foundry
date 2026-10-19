import pathlib,subprocess,json,datetime,hashlib,time
r=pathlib.Path('glm5-3-vllm/evidence/20261017-official-pd-optimization');ctl='/Users/wio/work/sglang-glm53-pd/ssh-control';ssh=['ssh','-S',ctl,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
def lifecycle(node,revision):
 ps=list(r.glob(f'node{node}-finite-window-node{node}-v{revision}-lifecycle-*/metrics.json'))
 if not ps or not (ps[0].parent/'result.json').exists():return None
 p=ps[0];b=p.read_bytes();m=json.loads((p.parent/'result.json').read_text());assert m['returncode']==0 and hashlib.sha256(b).hexdigest()==m['metrics.json']['sha256'];v=json.loads(b);assert not v['state']['Running'];return v
while not (lifecycle(32,11) and lifecycle(33,12)):
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,10,17,19,24,tzinfo=datetime.timezone.utc);time.sleep(2)
report={'client_started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exact owned proxy and reservation UIDs only, after both model lifecycles stopped. System/access pods and other jobs remain.','nodes':[],'reservation_actions':[]}
for node,pod,cid,revision in [(32,'sglang-node-access-32','d82b591171b15560f4524fcf9a380537be82c10ee1a6a77df495d01f77d5cffc',11),(33,'sglang-tuning-access-33-20261013','52c938a6f6d489b9256a8d75c73c46e4de451782b00469bbda89cc0fcc42fcf4',12)]:
 cfg={'node':node,'cid':cid,'revision':revision}
 code='''import pathlib,json,subprocess,hashlib,gzip,os
c=CONFIG;n=c['node'];assert os.uname().nodename=={32:'B4-33U-Atlas900RCKA3-32',33:'B5-5U-Atlas900RCKA3-33'}[n]
b=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd');d=json.loads(subprocess.check_output(['docker','inspect',c['cid']]))[0];assert not d['State']['Running'];assert d['Name']==f'/glm53-official-pd-{"prefill" if n==32 else "decode"}-{n}-v{c["revision"]}-20261017'
o={'node':n,'model_container_id':c['cid'],'model_stopped':True}
if n==32:
 p=json.loads((b/'proxy-v2/launch.json').read_text());cid=p['container_id'];assert cid=='46cf9a510fafdf41600fa8b5c092c27e5929bd7991c58f0fedcb12b96e1195ad';doc=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert doc['Name']=='/glm53-official-proxy-32-v2-20261017' and doc['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
 if doc['State']['Running']:subprocess.run(['docker','stop','-t','15',cid],check=True,capture_output=True,timeout=20)
 state=json.loads(subprocess.check_output(['docker','inspect',cid,'--format','{{json .State}}']));assert not state['Running'];logs=subprocess.run(['docker','logs','--timestamps',cid],text=False,capture_output=True,timeout=15);raw=logs.stdout+b'\\nSTDERR\\n'+logs.stderr;o['proxy']={'container_id':cid,'state':state,'raw_console_sha256':hashlib.sha256(raw).hexdigest()}
 out=b/'consumer-audits/node32-finite-proxy-stop-v2';out.mkdir(exist_ok=False);(out/'console.txt.gz').write_bytes(gzip.compress(raw,compresslevel=1));(out/'result.json').write_text(json.dumps(o,indent=2)+'\\n')
s=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert s.returncode==0;o['empty_cards']=s.stdout.count('No running processes found in NPU');o['npu_smi']=s.stdout+s.stderr;assert o['empty_cards']==8,'Compute process remains; preserve evidence and do not release reservation automatically';print(json.dumps(o))
'''.replace('CONFIG',repr(cfg))
 p=subprocess.run(ssh+['kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- python3 -'],input=code,text=True,capture_output=True,timeout=65);assert p.returncode==0,p.stderr;report['nodes'].append(json.loads(p.stdout))
for node,uid in [(32,'3b3ad135-298d-4a5d-96d1-b5070a34dc90'),(33,'6049992b-4112-4bd2-9578-0a2bbde40314')]:
 name=f'glm53-execution-reservation-{node}-20261017';p=subprocess.run(ssh+['kubectl get pod -n default '+name+' --ignore-not-found -o json'],capture_output=True,text=True,timeout=20)
 assert p.returncode==0,p.stderr
 if not p.stdout.strip():report['reservation_actions'].append({'node':node,'expected_uid':uid,'already_absent':True});continue
 v=json.loads(p.stdout);assert v['metadata']['uid']==uid and v['metadata']['labels']['owner']=='codex-execution-20261017' and not v['metadata'].get('ownerReferences');assert v['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}'
 d=subprocess.run(ssh+['kubectl delete pod -n default '+name+' --wait=false'],capture_output=True,text=True,timeout=20);assert d.returncode==0,d.stderr;report['reservation_actions'].append({'node':node,'name':name,'uid':uid,'original_active_deadline_seconds':v['spec'].get('activeDeadlineSeconds'),'delete_rc':d.returncode,'stdout':d.stdout})
deadline=time.monotonic()+25
while True:
 states=[]
 for node,uid in [(32,'3b3ad135-298d-4a5d-96d1-b5070a34dc90'),(33,'6049992b-4112-4bd2-9578-0a2bbde40314')]:
  name=f'glm53-execution-reservation-{node}-20261017';p=subprocess.run(ssh+['kubectl get pod -n default '+name+' --ignore-not-found -o json'],capture_output=True,text=True,timeout=15);assert p.returncode==0,p.stderr;v=json.loads(p.stdout) if p.stdout.strip() else None;states.append({'node':node,'owned_uid_absent':v is None or v['metadata']['uid']!=uid,'current_uid':v['metadata']['uid'] if v else None,'phase':v['status']['phase'] if v else None})
 if all(x['owned_uid_absent'] for x in states) or time.monotonic()>deadline:break
 time.sleep(2)
report.update(reservations_after=states,all_owned_reservations_released=all(x['owned_uid_absent'] for x in states),client_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest());(r/'finite-window-release-v1.json').write_text(json.dumps(report,indent=2)+'\n');(r/'finite-window-cleanup-source.py').write_text(pathlib.Path(__file__).read_text());print(json.dumps({k:report[k] for k in ['all_owned_reservations_released','reservations_after','client_finished_utc']}))
