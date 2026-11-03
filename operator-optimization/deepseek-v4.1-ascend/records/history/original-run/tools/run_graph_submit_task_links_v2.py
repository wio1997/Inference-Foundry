from pathlib import Path
import subprocess,shlex
T=Path(__file__).resolve().parents[1]
R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
src=T/'tools/verify_graph_submit_task_links_v2.py'
code='from pathlib import Path;p=Path('+repr(R+'/tools/'+src.name)+');assert not p.exists();p.write_text('+repr(src.read_text())+')'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',code])],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3',R+'/tools/'+src.name])],timeout=300);assert r.returncode==0,r.returncode
for name in ['summary.json','call_results.json']:
 code='from pathlib import Path;p=Path('+repr(R+'/graph-submit-task-link-validation-v2/'+name)+');assert p.stat().st_size<1024*1024;print(p.read_text(),end="")'
 r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',code])],capture_output=True,timeout=90);assert r.returncode==0,r.stderr.decode()
 p=T/'graph-submit-task-link-validation-v2'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(r.stdout)
