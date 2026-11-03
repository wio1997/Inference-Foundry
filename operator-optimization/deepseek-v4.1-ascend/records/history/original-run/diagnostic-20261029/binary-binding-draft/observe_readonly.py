import subprocess, shlex
code = r'''from pathlib import Path
import os,json,collections,re,datetime
p=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic/runtime')
print('observed_utc',datetime.datetime.now(datetime.timezone.utc).isoformat())
log=p/'logs/service.log'
if log.exists():
 print('service_log_bytes',log.stat().st_size,'service_log_modified_utc',datetime.datetime.fromtimestamp(log.stat().st_mtime,datetime.timezone.utc).isoformat())
 lines=collections.deque(log.open(errors='replace'),maxlen=250)
 keep=[line.strip()[:500] for line in lines if re.search(r'loading model weights|loading weights|Loading safetensors|model loading|model load|loading checkpoint|ERROR|Traceback|Exception|startup|EngineCore|Initialized|weight loading',line,re.I)]
 print('service_tail_filtered',json.dumps(keep[-18:],ensure_ascii=False))
else: print('service_log_absent')
for f in sorted((p/'metadata').glob('*')):
 if not f.is_file(): continue
 print('metadata_file',f.name,'bytes',f.stat().st_size,'mtime_utc',datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat())
 if f.name.startswith('binding-trace'):
  n=0; bad=0; counts=collections.Counter(); pids=set(); capped=0
  with f.open(errors='replace') as fp:
   for line in fp:
    try:
     row=json.loads(line);n+=1;counts[str(row.get('event','missing'))]+=1;pids.add(row.get('pid'));capped+=bool(row.get('record_cap_reached'))
    except (ValueError,TypeError): bad+=1
  print('binding_summary',json.dumps({'valid_records':n,'incomplete_or_invalid_lines':bad,'event_counts':dict(counts),'distinct_pids':len(pids),'record_cap_reached_records':capped},sort_keys=True))
'''
cmd=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code]
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(cmd)],capture_output=True,text=True,timeout=30)
print(r.stdout,end='');print(r.stderr,end='');raise SystemExit(r.returncode)
