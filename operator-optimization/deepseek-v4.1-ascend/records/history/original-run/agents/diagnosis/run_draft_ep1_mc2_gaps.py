#!/usr/bin/env python3
"""Dispatch fixed CPU-only historical query in the dedicated container."""
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2]; D=T/'agents/diagnosis'
inp=D/'draft-six-mc2-preceding-csv.json'; query=D/'query_draft_ep1_mc2_gaps.py'; out=D/'draft-ep1-mc2-gaps-all-streams.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if sha(inp)!='5de5efc398e82f21c28f1aabde8bd88141afb3343bf42d52289eb2126cdb7c42' or out.exists():
    raise RuntimeError('input mismatch or output exists')
d=json.loads(inp.read_text()); rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt'; windows=[]
for sample in d['samples']:
    if sample['rank']!=rank: continue
    if len(sample['latest_end_rows'])!=1 or sample['latest_end_rows'][0]['Type']!='MoeGatingTopKHash': raise RuntimeError('unexpected predecessor')
    windows.append({'op_id':sample['local_op_id'],'start_ns':sample['latest_end_rows'][0]['end_ns'],
      'end_ns':sample['target']['start_ns'],'main_stream':int(sample['target']['Stream ID'])})
spec={'rank':rank,'windows':windows,'csv_sha256':d['csv_sha256_by_rank'][rank],'db_sha256':d['db_sha256_by_rank'][rank]}
code=query.read_text().replace('FROZEN_INPUT',repr(json.dumps(spec,separators=(',',':'))))
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'; container='wio-vllm-v41-repro169-stage-20261028'
trace='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([ssh,shlex.join(['docker','exec',container,'python3','-c',code,'--trace-root',trace])],capture_output=True,text=True,timeout=180)
if r.returncode: raise RuntimeError('query failed: '+r.stderr[-2000:])
result=json.loads(r.stdout);result['input_sha256']=sha(inp);result['query_sha256']=sha(query);result['wrapper_sha256']=sha(Path(__file__))
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'sha256':sha(out)}))
