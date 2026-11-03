from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_layer0_1_gap_events.py'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
D='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',S.read_text(),'--task-root',R,'--trace-root',D])],capture_output=True,text=True,timeout=240)
assert r.returncode==0,r.stderr
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(S.read_bytes()).hexdigest();x['no_device_work']=True
out=T/'agents/diagnosis/engram-boundary-controls.json';assert not out.exists();out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'summary':x['summary']}))
