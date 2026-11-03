#!/usr/bin/env python3
import subprocess,shlex,pathlib
p=pathlib.Path(__file__).resolve().parent
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
code="from pathlib import Path; import json; root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run'); [print(d,json.dumps([x for x in json.loads((root/d/'input-copy-manifest.json').read_text()) if x['relative_path'].endswith(('.db','trace_view.json'))])) for d in ('offline-ep8-20261029','offline-ep8-text-20261029')]; [print(d, 'trace_files',len(list((root/d).rglob('trace_view.json')))) for d in ('offline-ep8-20261029','offline-ep8-text-20261029')]"
args=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code]
r=subprocess.run([ssh,shlex.join(args)],capture_output=True,text=True,check=True)
(p/'ep8-original-export-presence.txt').write_text(r.stdout)
print('recorded manifest presence')
