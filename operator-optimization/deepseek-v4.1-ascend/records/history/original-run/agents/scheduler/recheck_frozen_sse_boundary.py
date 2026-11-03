#!/usr/bin/env python3
"""Offline frozen two-round SSE-window and counter audit; do not print predictions."""
from pathlib import Path
from fractions import Fraction
import hashlib, io, json, sqlite3
import numpy as np
root=Path(__file__).resolve().parents[2]
raw=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results')
package=Path('/Users/wio/work/173-sglang-matched-172-benchmark/package/tools/aisbench-7bcb6440da11')
manifest=json.loads((raw/'client-source-SHA256.json').read_text())
source_paths={
 'prefix_api':('ais_bench_prefix_cache/models/vllm_prefix_cache_api.py','plugins/prefix_cache/ais_bench_prefix_cache/models/vllm_prefix_cache_api.py'),
 'vllm_api':('ais_bench/benchmark/models/api_models/vllm_custom_api.py','ais_bench/benchmark/models/api_models/vllm_custom_api.py'),
 'output':('ais_bench/benchmark/models/output.py','ais_bench/benchmark/models/output.py'),
 'perf_summary':('ais_bench/benchmark/summarizers/default_perf.py','ais_bench/benchmark/summarizers/default_perf.py'),
 'perf_calc':('ais_bench/benchmark/calculators/default_perf_metric_calculator.py','ais_bench/benchmark/calculators/default_perf_metric_calculator.py')}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
for name,(key,rel) in source_paths.items():
 p=package/rel; actual=sha(p)
 if actual!=manifest.get(key):raise RuntimeError('CLIENT_SOURCE_HASH_MISMATCH: '+name)
 sources[name]={'sha256':actual,'local_path':str(p),'manifest_key':key}
bound=json.loads((root/'evidence/fixed-output-conditional-bound.json').read_text())
results_hash=json.loads((raw/'RESULT-SHA256.json').read_text())
rounds=[]
def counter(p):
 total=0.0
 for line in p.read_text().splitlines():
  if line.startswith('vllm:generation_tokens_total{') or line.startswith('vllm:generation_tokens_total '):
   total+=float(line.rsplit(' ',1)[1])
 return total
for phase,s in zip(('measure1','measure2'),bound['samples']):
 path=Path(s['input_path']);data=path.read_bytes()
 if hashlib.sha256(data).hexdigest()!=s['sha256']:raise RuntimeError('DETAILS_HASH_MISMATCH')
 rows=[json.loads(x) for x in data.splitlines() if x.strip()]
 if len(rows)!=8 or not all(x['success'] for x in rows):raise RuntimeError('RESULT_ROWS_NOT_EIGHT_SUCCESS')
 token_count=sum(int(x['output_tokens']) for x in rows); length=sum(len(x['prediction'].encode('utf-8')) for x in rows)
 if token_count!=8192 or length!=s['total_text_utf8_bytes']:raise RuntimeError('COUNT_MISMATCH')
 dbnames={x['db_name'] for x in rows}
 if len(dbnames)!=1:raise RuntimeError('MULTIPLE_DATABASES')
 name=next(iter(dbnames)); dbpath=path.parent/'db_data'/name
 expected=results_hash.get(str(dbpath.relative_to(raw)))
 actual=sha(dbpath)
 if expected!=actual:raise RuntimeError('DATABASE_HASH_MISMATCH')
 refs=[x['time_points']['__db_ref__'] for x in rows]
 if len(set(refs))!=8:raise RuntimeError('DUPLICATE_TIME_REFERENCE')
 con=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True);times=[]
 for ref in refs:
  result=con.execute('SELECT arr_blob FROM numpy_store WHERE id=?',(ref,)).fetchone()
  if not result:raise RuntimeError('MISSING_TIMING_ARRAY')
  a=np.load(io.BytesIO(result[0]),allow_pickle=False)
  if a.ndim!=1 or a.size<2 or not np.all(np.isfinite(a)) or not np.all(a[1:]>=a[:-1]):raise RuntimeError('INVALID_TIMING_ARRAY')
  times.append(a)
 con.close(); elapsed=float(max(a[-1] for a in times)-min(a[0] for a in times))
 progress=json.loads((raw/'results'/phase/'progress.json').read_text())
 if abs(elapsed-float(progress['duration_seconds']))>1e-6:raise RuntimeError('CLIENT_WINDOW_MISMATCH')
 delta=counter(raw/'results'/phase/'metrics-after.txt')-counter(raw/'results'/phase/'metrics-before.txt')
 if delta!=8192 or progress['server_counter_deltas']['generation_tokens_total']!=delta:raise RuntimeError('SERVER_COUNTER_MISMATCH')
 q_per_d=Fraction(token_count,length)
 rounds.append({'phase':phase,'details_sha256':s['sha256'],'database_sha256':actual,'success_requests':len(rows),
  'output_tokens_recorded':token_count,'prediction_utf8_bytes':length,
  'client_window_seconds_from_saved_timepoints':elapsed,'client_window_seconds_saved_rounded':progress['duration_seconds'],
  'per_request_timepoints_minus_one':[int(a.size-1) for a in times],
  'server_generation_counter_delta':delta,'token_per_logical_byte_numerator':q_per_d.numerator,'token_per_logical_byte_denominator':q_per_d.denominator})
out={'status':'ARITHMETIC_AND_HASH_CHECK_OK','recheck_script_sha256':sha(Path(__file__)),'checked_client_source_manifest_sha256':sha(raw/'client-source-SHA256.json'),
 'checked_result_manifest_sha256':sha(raw/'RESULT-SHA256.json'),'sources':sources,'rounds':rounds,
 'model_scope':'historical completions SSE client and recorded fixed predictions under unmodified runtime path',
 'boundary':'aiohttp optionally decompressed SSE stream to application raw_chunk iterator, aggregated across requests',
 'not_proven':['actual runtime import/object unmodified','actual raw SSE bytes or compression state','server counter interval identical to client time interval',
 'one server token per output chunk','HBM/network physical traffic','physical B_decoded_max','universal model or NPU TPS upper bound'],
 'no_npu_calls':True}
p=root/'agents/scheduler/frozen-sse-boundary-check.json'
if p.exists():raise RuntimeError('STOP_OUTPUT_ALREADY_EXISTS')
p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'result':str(p),'rounds':rounds},ensure_ascii=False))
