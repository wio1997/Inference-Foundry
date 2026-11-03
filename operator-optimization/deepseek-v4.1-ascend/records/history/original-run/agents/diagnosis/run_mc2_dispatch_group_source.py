from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/read_mc2_dispatch_group_source.py';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';cs=['wio-vllm-v41-repro169-20261028','wio-vllm-v41-repro169-stage-20261028'];out=[]
for c in cs:
 r=subprocess.run([SSH,shlex.join(['docker','exec',c,'python3','-c',S.read_text()])],capture_output=True,text=True,timeout=180)
 if r.returncode:out.append({'container':c,'status':'UNAVAILABLE','error_sha256':hashlib.sha256(r.stderr.encode()).hexdigest()});continue
 out.append({'container':c,'status':'READ','data':json.loads(r.stdout)})
p=T/'agents/diagnosis/mc2-dispatch-group-source.json'
if p.exists():raise RuntimeError('output exists')
p.write_text(json.dumps({'containers':out,'script_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'no_device_work':True},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(p),'containers':[{'container':q['container'],'status':q['status']} for q in out]}))
