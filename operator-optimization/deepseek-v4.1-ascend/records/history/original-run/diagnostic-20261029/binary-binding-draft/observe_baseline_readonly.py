import subprocess,shlex
code=r'''from pathlib import Path
import collections,json,datetime,re
p=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/runtime/logs')
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'directory_exists':p.exists(),'files':[]}
patterns={'weight_finished':r'Loading weights took|Loading model weights took','draft_loaded':r'DSpark draft model loaded','service_ready':r'Application startup complete|Uvicorn running|Started server process|EngineCore.*initialized','error':r'\bERROR\b|Traceback|Exception','loading':r'Loading safetensors|Loading checkpoint|Loading weights|Loading model weights'}
if p.exists():
 for f in sorted(p.glob('*')):
  if not f.is_file() or f.suffix not in ('.log','.txt'):continue
  st=f.stat();matches=collections.Counter();last={}
  with f.open(errors='replace') as stream:
   tail=collections.deque(stream,maxlen=2000)
  for line in tail:
   for k,v in patterns.items():
    if re.search(v,line,re.I):
     matches[k]+=1
     t=re.search(r'\b\d\d-\d\d \d\d:\d\d:\d\d\b',line)
     if t:last[k]=t.group()
  out['files'].append({'name':f.name,'bytes':st.st_size,'mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'tail_lines_examined':len(tail),'keyword_counts':dict(matches),'last_matching_timestamps':last})
print(json.dumps(out,sort_keys=True))
'''
cmd=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code]
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(cmd)],capture_output=True,text=True,timeout=30)
print(r.stdout,end='');print(r.stderr,end='');raise SystemExit(r.returncode)
