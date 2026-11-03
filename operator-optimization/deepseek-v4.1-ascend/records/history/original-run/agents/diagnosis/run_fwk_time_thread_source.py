from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/read_fwk_time_thread_source.py';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',S.read_text()])],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only source query failed: '+r.stderr)
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(S.read_bytes()).hexdigest();x['no_device_work']=True
out=T/'agents/diagnosis/fwk-time-thread-source.json'
if out.exists():raise RuntimeError('output exists')
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'source_files':len(x['files'])}))
