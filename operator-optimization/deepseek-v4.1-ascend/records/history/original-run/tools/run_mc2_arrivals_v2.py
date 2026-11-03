from pathlib import Path
import subprocess,shlex
T=Path(__file__).resolve().parents[1];R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';STAGE='wio-vllm-v41-repro169-stage-20261028'
code='from pathlib import Path;p=Path('+repr(R+'/tools/analyze_mc2_arrivals_v2.py')+');assert not p.exists();p.write_text('+repr((T/'tools/analyze_mc2_arrivals_v2.py').read_text())+')'
p=subprocess.run([SSH,shlex.join(['docker','exec',STAGE,'python3','-c',code])],capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
p=subprocess.run([SSH,shlex.join(['docker','exec',STAGE,'python3',R+'/tools/analyze_mc2_arrivals_v2.py'])],text=True,timeout=180);raise SystemExit(p.returncode)
