import subprocess,shlex
code=r'''from pathlib import Path
import hashlib,json,collections,datetime
p=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic/runtime')
f=p/'profile-client-results/progress.json'
b=f.read_bytes();r=json.loads(b)
keys=['state','updated_at','not_a_benchmark_score','profile_max_iterations','case','profile_start_http','profile_stop_http','warmup_case_valid','warmup_counter_deltas','measure1_case_valid','measure1_counter_deltas','exact_frozen_prediction_matches']
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'progress_sha256':hashlib.sha256(b).hexdigest(),'progress_bytes':len(b),'progress':{k:r[k] for k in keys if k in r},'error_recorded':bool(r.get('error') or r.get('traceback'))}
roots=[x for x in (p/'diagnostic-traces').glob('*_ascend_pt') if x.is_dir()]; ext=collections.Counter(); bytecounts=collections.Counter(); info=0; dirs=collections.Counter();csv=0;views=0
for root in roots:
 for f in root.rglob('*'):
  if not f.is_file():continue
  relative=f.relative_to(root); top=relative.parts[0];dirs[top]+=1
  suffix=f.suffix or '<none>'; ext[suffix]+=1;bytecounts[top]+=f.stat().st_size
  info+=int(f.name.startswith('profiler_info') and f.suffix=='.json')
  csv+=int(f.name=='kernel_details.csv');views+=int(f.name=='trace_view.json')
out['profile']={'worker_root_count':len(roots),'profiler_info_count':info,'file_count_by_root_child':dict(dirs),'bytes_by_root_child':dict(bytecounts),'file_count_by_extension':dict(ext),'kernel_details_count':csv,'trace_view_count':views}
print(json.dumps(out,sort_keys=True))
'''
cmd=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code]
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(cmd)],capture_output=True,text=True,timeout=45)
print(r.stdout,end='');print(r.stderr,end='');raise SystemExit(r.returncode)
