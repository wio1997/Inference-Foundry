from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/read_ep_group_source.py';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';cs=['wio-vllm-v41-repro169-20261028','wio-vllm-v41-repro169-stage-20261028'];results=[]
for c in cs:
 r=subprocess.run([SSH,shlex.join(['docker','exec',c,'python3','-c',S.read_text()])],capture_output=True,text=True,timeout=180)
 if r.returncode:results.append({'container':c,'status':'UNAVAILABLE','error_sha256':hashlib.sha256(r.stderr.encode()).hexdigest()});continue
 results.append({'container':c,'status':'READ','data':json.loads(r.stdout)})
out=T/'agents/diagnosis/ep-group-source.json'
if out.exists():raise RuntimeError('output exists')
out.write_text(json.dumps({'containers':results,'script_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'no_device_work':True},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'container_statuses':[{'container':r['container'],'status':r['status']} for r in results]}))
