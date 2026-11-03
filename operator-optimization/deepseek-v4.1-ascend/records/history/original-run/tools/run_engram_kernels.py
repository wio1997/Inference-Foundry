from pathlib import Path
import subprocess,shlex
T=Path(__file__).resolve().parents[1];R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028';src=T/'tools/inspect_engram_kernels.py'
code='from pathlib import Path;p=Path('+repr(R+'/tools/'+src.name)+');assert not p.exists();p.write_text('+repr(src.read_text())+')'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',code])],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3',R+'/tools/'+src.name])],timeout=360);assert r.returncode==0,r.returncode
code='from pathlib import Path;p=Path('+repr(R+'/engram-kernel-profile/summary.json')+');assert p.stat().st_size<100000;print(p.read_text(),end="")'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',code])],capture_output=True,timeout=90);assert r.returncode==0,r.stderr.decode()
p=T/'engram-kernel-profile/summary.json';p.parent.mkdir(exist_ok=True);p.write_bytes(r.stdout)
