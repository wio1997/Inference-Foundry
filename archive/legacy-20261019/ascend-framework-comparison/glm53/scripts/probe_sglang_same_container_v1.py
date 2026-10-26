#!/usr/bin/env python3
"""Query NPU from an existing owned pool container; no new device mount or init."""
import ast,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();dep=P.parent/'probe_sglang_driver_library_v2.py'
assert hashlib.sha256(dep.read_bytes()).hexdigest()=='25a0f7eb93ce43575d5ca7dafb0c4a403773a9d3c9804605ffbe1aafb9291a51'
s=importlib.util.spec_from_file_location('driver_probe',dep);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RUN=m.base.b.ROOT/'runs/20261017T1690-sglang-same-container-query-v1'
HOST=r'''import subprocess,json,re
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/glm53-sglang-four-node-pool-v1-20261017-34' and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1645-sglang-four-node-pool-start-v1' and d['State']['Running']
pid=d['State']['Pid']
def npu():
 r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);rows=re.findall(r'^|s*(d+)s+(d+)s*|s*(d+)s*|',r.stdout.split('Process id',1)[-1],re.M)
 assert r.returncode==0 and {int(x[2]) for x in rows}<={pid} and len({x[0] for x in rows})+r.stdout.count('No running processes')==8
 return r.stdout
before=npu();r=subprocess.run(['docker','exec',CID,'python3','-c',CODE],capture_output=True,text=True,timeout=100)
print(json.dumps({'container_id':CID,'query_rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'npu_before':before,'npu_after':npu(),'model_requests':0,'SDK_init_calls':0}))
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 code=m.base.replace(HOST,{'CID':m.PROVIDER,'IMAGE':m.IMAGE,'CODE':m.CODE});ast.parse(code)
 result=m.base.invoke(34,code,timeout=200);(RUN/'node-34.json').write_text(json.dumps(result,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':result['rc'],'model_requests':0,'SDK_init_calls':0},indent=2)+'\n')
if __name__=='__main__':main()
