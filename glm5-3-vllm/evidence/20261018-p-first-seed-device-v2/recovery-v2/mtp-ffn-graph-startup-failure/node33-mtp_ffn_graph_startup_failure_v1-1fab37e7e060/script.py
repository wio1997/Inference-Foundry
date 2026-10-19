from pathlib import Path
import gzip
import json
import shutil
import subprocess

cid='1bb3a19d7f6ed621b5fcf72189bbfa85c9a3ecb46753706ae5bf7cd1743a1ad2'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live1-20261018/node33')
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['Id']==cid and d['Name']=='/glm53-mtp-ffn-graph-33-live1-20261018'
assert d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
assert not d['State']['Running'] and d['State']['ExitCode']==1
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
for name in ['plan.json','launch.json','context.json','manifest.json']:
    shutil.copyfile(root/name,out/name)
q=subprocess.run(['docker','logs','--timestamps',cid],capture_output=True,timeout=20)
b=q.stdout+b'\n'+q.stderr
with gzip.open(out/'failed-console.txt.gz','wb') as f:f.write(b)
lines=b.decode(errors='replace').splitlines()
cause=[x for x in lines if '(Worker_TP8 ' in x and any(z in x for z in ['Dynamo does not know','Unsupported method','ContextVar','glm53-seed-mtp_ffn_graph'])]
assert cause
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=10)
assert smi.count('No running processes found in NPU')==8
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'state':d['State'],'cause':cause,'NPU_idle':True,'console_bytes':len(b),'candidate_requests_executed':0,'P_and_proxy_kept':True,'scope':'Failed first prototype globallywrappedMoE leaked ContextVar into target Dynamo compile. No ordinary request or new graph capture reached. Exact failed model naturally exited; full console/config retained.'})+'\nGLM53_RESULT_JSON_END')
