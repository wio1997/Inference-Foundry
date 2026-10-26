"""Render exact-container cache-gate stage/invoke code; no implicit resources."""
import ast,base64,hashlib,json,re

STAGE=r'''import pathlib,subprocess,json,hashlib,base64
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and not d['HostConfig']['Privileged']
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED_RID
shared=pathlib.Path(SHARED_ROOT);supervisor=json.loads((shared/'supervisor-status.json').read_text());provider=json.loads((shared/'provider-status.json').read_text())
assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] and 'model_exit_code' not in supervisor and provider['phase']=='READY'
parent=shared/'formal-gates'/RID;data=parent/'data';assert data.is_dir()
assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()==MANIFEST_SHA
control=parent/'control';assert not control.exists(),'No gate-stage overwrite/replay';control.mkdir()
for name,item in FILES.items():
 assert pathlib.Path(name).name==name
 raw=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['sha256'];(control/name).write_bytes(raw)
print(json.dumps({'project_id':'glm53','framework':FRAMEWORK,'node':NODE,'container_id':CID,'control':str(control),'files':{name:item['sha256'] for name,item in FILES.items()},'model_requests':0,'SDK_init_calls':0}))
'''
INVOKE=r'''import pathlib,subprocess,json,hashlib,time
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running']
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED_RID
shared=pathlib.Path(SHARED_ROOT);state=json.loads((shared/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and state['model_started'] and 'model_exit_code' not in state
control=shared/'formal-gates'/RID/'control'
for name,item in FILES.items():assert hashlib.sha256((control/name).read_bytes()).hexdigest()==item['sha256']
assert REQUEST['framework']==FRAMEWORK and REQUEST['namespace']==NAMESPACE and REQUEST['concurrency']==CONCURRENCY and REQUEST['manifest_sha256']==MANIFEST_SHA
with (control/'gate-controller.once').open('x') as f:f.write(NAMESPACE)
(control/'need-cache-gate.json').write_text(json.dumps(REQUEST)+'\n')
container_root='/run/glm53/formal-gates/'+RID+'/control'
settings_raw=(control/'gate-settings.json').read_bytes()
boot='set -eu; export PYTHONPATH='+container_root+':'+SDK_PREFIX+'/opt/glm53/source/python:${PYTHONPATH:-}; '+LIB_PREFIX+'exec python3 '+container_root+'/glm53_formal_cache_gate_v1.py --settings '+container_root+'/gate-settings.json --sha256 '+hashlib.sha256(settings_raw).hexdigest()
with (control/'gate-stdout.txt').open('x') as stdout,(control/'gate-stderr.txt').open('x') as stderr:
 process=subprocess.Popen(['docker','exec',CID,'/bin/bash','-c',boot],stdout=stdout,stderr=stderr,start_new_session=True)
 (control/'gate-host-pid.json').write_text(json.dumps({'pid':process.pid,'start_monotonic_ns':time.monotonic_ns()}))
 try:rc=process.wait(timeout=240)
 except subprocess.TimeoutExpired:
  print(json.dumps({'project_id':'glm53','framework':FRAMEWORK,'namespace':NAMESPACE,'prefix_count':CONCURRENCY,'pass':False,'all_expected_keys_present':False,'actual_payload_samples_read':False,'close_rc':None,'error':'SDK gate deadline exceeded; owned child preserved, no formal dispatch allowed','host_pid':process.pid}));raise SystemExit(1)
report=control/'cache-gate.json'
assert report.is_file(),('SDK gate did not produce report',rc)
raw=report.read_bytes();assert len(raw)<=8*1024**2
payload=json.loads(raw);assert payload['project_id']=='glm53' and payload['framework']==FRAMEWORK and payload['namespace']==NAMESPACE
print(json.dumps({'query_rc':rc,'report_sha256':hashlib.sha256(raw).hexdigest(),'report':payload}))
raise SystemExit(rc)
'''

def render(template,values):
    pattern=r'\b(?:'+'|'.join(re.escape(key) for key in values)+r')\b'
    text=re.sub(pattern,lambda match:repr(values[match.group(0)]),template)
    ast.parse(text);return text
