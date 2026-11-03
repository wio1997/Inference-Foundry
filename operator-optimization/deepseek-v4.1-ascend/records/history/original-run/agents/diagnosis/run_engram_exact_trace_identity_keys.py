#!/usr/bin/env python3
"""Run fixed, redacted trace-args query in dedicated CPU staging container."""
import hashlib,json,shlex,subprocess
from decimal import Decimal
from pathlib import Path
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_engram_exact_trace_identity_keys.py';A=T/'agents/diagnosis/ep8-trace-event-args.json';OUT=T/'agents/diagnosis/engram-exact-trace-identity-keys.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028';TRACE='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json'
script=S.read_bytes();qsha=hashlib.sha256(script).hexdigest()
if qsha!='d94f311ca19fecdd4eb9e8cc98fe708b0397d4fef98583c793699b51a68808e6':raise RuntimeError('query SHA mismatch')
b=A.read_bytes();asha=hashlib.sha256(b).hexdigest()
if asha!='d2e9c8ee7ce178c106dbb02d7b3aae1bf59dc1f65206607e971ed65d156a7959':raise RuntimeError('stage121 SHA mismatch')
x=json.loads(b)
if x['trace_sha256']!='3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99':raise RuntimeError('stage121 trace SHA mismatch')
expected={(2,1793152960185069105,53682),(2,1793152960185562115,53689),(2,1793152960187571215,72),(2,1793152960198990467,730),(3,1793152960260742113,55039),(3,1793152960261206424,55046),(3,1793152960263910410,72),(3,1793152960275488198,730)};actual=set()
for row in x['targets']:
 t=row['target']
 if t['run'] not in (2,3) or t['kind'] not in ('MatMulV2','_engram_host_uva_gather_dequant_kernel'):continue
 z=[v for v in row['nearby_trace_events'] if v['phase']=='X' and v['pid']==744904 and v['tid']==t['stream'] and int(Decimal(v['timestamp_us'])*1000)==t['start_ns'] and 'Task Id' in v['numeric_identifier_args']]
 if len(z)!=1 or row['truncated']:raise RuntimeError('stage121 anchor ambiguous')
 actual.add((t['run'],t['start_ns'],z[0]['numeric_identifier_args']['Task Id']))
if actual!=expected:raise RuntimeError('stage121 anchors differ')
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--trace',TRACE]);r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=300)
if r.returncode:raise RuntimeError('query failed exit '+str(r.returncode))
y=json.loads(r.stdout)
if len(y['samples'])!=8 or y['trace_sha256']!=x['trace_sha256']:raise RuntimeError('unexpected result')
y['stage121_sha256']=asha;y['query_sha256']=qsha;y['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();y['no_device_work']=True
body=(json.dumps(y,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_TRACE_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),'matched':sum(s['match_count']==1 for s in y['samples'])}))
