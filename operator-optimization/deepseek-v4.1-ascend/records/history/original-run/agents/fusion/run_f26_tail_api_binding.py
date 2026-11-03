#!/usr/bin/env python3
"""Run F26 bounded SQLite audit on existing 169 stage container; no device load."""
import base64,pathlib,shlex,subprocess
root=pathlib.Path(__file__).resolve().parents[2]
code=(root/'agents/fusion/query_f26_tail_api_binding.py').read_bytes();enc=base64.b64encode(code).decode()
base='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
db='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
py='import base64,sys;sys.argv='+repr(['query.py','--task-root',base,'--db',db])+';exec(compile(base64.b64decode('+repr(enc)+'),"query.py","exec"))'
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',py]);p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=50,check=True)
(root/'agents/fusion/f26-tail-api-binding.json').write_text(p.stdout)
print('Saved bounded F26 API binding audit')
