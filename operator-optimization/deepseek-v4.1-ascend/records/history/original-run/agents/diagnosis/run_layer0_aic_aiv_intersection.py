#!/usr/bin/env python3
"""Run reviewed fixed query in existing staging container; no device activity."""
from pathlib import Path
from hashlib import sha256
import json,shlex,subprocess
T=Path(__file__).resolve().parents[2]
S=T/'agents/diagnosis/query_layer0_aic_aiv_intersection.py'
EXPECTED='d99a37b5998ab7c8a8ac7a317be1ef4785c20b6b61c6fc6af241dfed88526ff9'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
out=T/'agents/diagnosis/layer0-aic-aiv-intersection.json'
if sha256(S.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('STOP query script identity')
if out.exists():raise RuntimeError('STOP output exists')
cmd=shlex.join(['docker','exec',C,'python3','-c',S.read_text()])
r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=240)
if r.returncode!=0:
 known=('fixed input hash mismatch','manifest coverage','rank coverage','bad rank','window coverage','window model or overlap','CSV identity','unexpected task type','positive-cycle ratio invalid','CSV changed','per-window attribution mismatch','layer0 count mismatch','counter summary mismatch','cycle count mismatch')
 found=[v for v in known if ('RuntimeError: '+v) in r.stderr]
 import re
 frames=re.findall(r'File "<string>", line ([0-9]+)',r.stderr)
 classes=[c for c in ('KeyError','ValueError','TypeError','AttributeError','IndexError','OSError','RuntimeError','AssertionError','FileNotFoundError','NameError') if re.search(r'(?m)^'+c+r'(?::|$)',r.stderr)]
 detail=(found[0] if len(found)==1 else 'unclassified')+'; exception='+(classes[0] if len(classes)==1 else 'unknown')+'; line='+(frames[-1] if frames else 'unknown')
 raise RuntimeError('STOP remote query failed: '+detail)
if len(r.stdout)>100000:raise RuntimeError('STOP oversized response')
x=json.loads(r.stdout)
if x.get('rows')!=1824 or x.get('coverage_windows')!=48 or sum(g['rows'] for g in x['groups'])!=1824 or sum(t['rows'] for t in x['group_by_type'])!=1824:raise RuntimeError('STOP incomplete result')
x['query_script_sha256']=EXPECTED
with out.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':'QUERY_COMPLETED','windows':48,'rows':1824,'sha256':sha256(out.read_bytes()).hexdigest()}))
