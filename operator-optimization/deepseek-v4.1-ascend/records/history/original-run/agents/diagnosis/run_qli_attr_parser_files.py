#!/usr/bin/env python3
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];q=T/'agents/diagnosis/query_qli_attr_parser_files.py';out=T/'agents/diagnosis/qli-attr-parser-files.json';ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';c='wio-vllm-v41-repro169-stage-20261028'
if out.exists():raise RuntimeError('output exists')
r=subprocess.run([ssh,shlex.join(['docker','exec',c,'python3','-c',q.read_text()])],capture_output=True,text=True,timeout=120)
if r.returncode:raise RuntimeError('read-only query failed '+r.stderr[-1500:])
x=json.loads(r.stdout);x['query_sha256']=hashlib.sha256(q.read_bytes()).hexdigest();x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'OK','output':str(out),'files':len(x['files'])}))
