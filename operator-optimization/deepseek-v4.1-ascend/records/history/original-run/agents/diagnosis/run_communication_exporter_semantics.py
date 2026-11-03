from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/locate_communication_exporter_semantics.py'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
args=['docker','exec',C,'python3','-c',S.read_text()]
r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('remote read-only source inventory failed: '+r.stderr)
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(S.read_bytes()).hexdigest();x['no_device_work']=True
out=T/'agents/diagnosis/communication-exporter-source-inventory.json'
if out.exists():raise RuntimeError('Output already exists; refusing overwrite')
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'files_scanned':x['files_scanned'],'matching_files':len(x['matches'])}))
