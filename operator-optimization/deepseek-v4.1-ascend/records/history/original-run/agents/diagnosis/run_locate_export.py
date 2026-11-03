from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2]; S=T/"agents/diagnosis/locate_pmu_export_source.py"; SSH="/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh"; C="wio-vllm-v41-repro169-stage-20261028"
r=subprocess.run([SSH,shlex.join(["docker","exec",C,"python3","-c",S.read_text()])],capture_output=True,text=True,timeout=180);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);x["script_sha256"]=hashlib.sha256(S.read_bytes()).hexdigest();x["no_device_work"]=True
out=T/"agents/diagnosis/pmu-export-source-locations.json";assert not out.exists();out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n");print(json.dumps({"scanned":x["files_scanned"],"matches":len(x["results"]),"skipped":len(x["skipped"])}))
