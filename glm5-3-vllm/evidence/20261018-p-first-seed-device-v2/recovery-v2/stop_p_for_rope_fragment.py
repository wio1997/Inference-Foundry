from pathlib import Path
import hashlib,json,subprocess,sys
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
p=next((r/'p-bucket-control-snapshot').glob('*/artifacts-backup.json'));x=json.loads(p.read_text());b=Path(x['local_archive']).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
assert json.loads((r/'p-bucket-control-outcome-v1.json').read_text())['completed']
code="""import json,subprocess,time,urllib.request
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 z=[v for v in s.splitlines() if v.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(z)>=3 and all(float(v.rsplit(' ',1)[1])==0 for v in z)
expected=[('5c38dc4d5ea985fccddf7305173f49e91372f6ebcd29bce5ff36fe3aeb76426c','glm53-official-proxy-32-p-bucket-control-v1-20261019'),('ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758','glm53-p-bucket-control-32-v1-20261019')];rows=[]
for cid,name in expected:
 d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['Id']==cid and d['Name']=='/'+name and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83';before=d['State']
 if before['Running']:subprocess.run(['docker','stop','-t','30',cid],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 after=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State'];assert not after['Running'];rows.append({'CID':cid,'before':before,'after':after})
for i in range(20):
 s=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
 if s.count('No running processes found in NPU')==8:break
 time.sleep(1)
assert s.count('No running processes found in NPU')==8
print(json.dumps({'stopped_exact_owned':rows,'NPU_idle':s,'D_unchanged':True}))
"""
a=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34','kubectl exec -i -n default sglang-node-access-32 -- nsenter -t 1 -m -- python3 -'];q=subprocess.run(a,input=code,text=True,capture_output=True,timeout=150);assert q.returncode==0,q.stdout+q.stderr;x=json.loads(q.stdout);(r/'p-bucket-control-final-stop-for-fragment-v1.json').write_text(json.dumps(x,indent=2)+'\n');print('P_and_proxy_stopped_exactly; node32 NPU idle; D kept')
