from pathlib import Path
import subprocess,shlex,json,hashlib
p=Path('agents/diagnosis/16rank-engram-matmul-metrics.json');raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d'
rows=[s['target'] for s in json.loads(raw)['samples'] if 'ep8_rank0_727' in s['target']['rank']]
assert len(rows)==6 and len({(x['run'],x['after_layer']) for x in rows})==6
code="import pathlib,hashlib,sqlite3,json\np=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db')\nh=hashlib.sha256()\nwith p.open('rb') as f:\n for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)\nassert h.hexdigest()=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'\nc=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)\nrows=c.execute('select deviceId,timestampNs,freq,dieId from AICORE_FREQ order by timestampNs').fetchall();c.close()\nprint(json.dumps({'db_sha256':h.hexdigest(),'freq_rows':rows}))\n"
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=60)
if r.returncode:raise RuntimeError('read-only query failed '+r.stderr[-500:])
data=json.loads(r.stdout);freq=data['freq_rows']
out=[]
for t in rows:
 before=[q for q in freq if q[1]<=t['start_ns']]
 after=[q for q in freq if q[1]>=t['end_ns']]
 within=[q for q in freq if t['start_ns']<q[1]<t['end_ns']]
 out.append({'run':t['run'],'after_layer':t['after_layer'],'last_prior_freq':before[-1][2] if before else None,'next_freq':after[0][2] if after else None,'inside_frequency_rows':len(within),'prior_gap_us':(t['start_ns']-before[-1][1])/1000 if before else None,'next_gap_us':(after[0][1]-t['end_ns'])/1000 if after else None})
print(json.dumps({'db_sha256':data['db_sha256'],'freq_rows':freq,'task_count':len(rows),'tasks':out,'scope':'EP8 only, DB exported timestamp comparisons; no raw decode, clock/export completeness not proven'},indent=2))
