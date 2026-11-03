import subprocess,shlex
code=r'''from pathlib import Path
import json,hashlib,datetime,collections
p=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic/runtime/metadata/binding-trace.jsonl')
known={'4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9','d88acd30e3b56893952a9e7094dc4c2c45ceed08942383f9e9d4c9dd3e2260b5','a0aec99d7145b29bb226231ff24d751d0798ff5ead7a2be84cc5db0632797979'}
with p.open('rb') as f:
 st=f.stat() if hasattr(f,'stat') else __import__('os').fstat(f.fileno()); data=f.read(min(st.st_size,16777216))
counts=collections.Counter();bad=0;reported_drops=0;event_losses=0;found=[];active={};funcs={};caps=[];seqcap=[];coverage=collections.Counter();snapshot_incomplete=bool(data and not data.endswith(b'\n'))
for i,line in enumerate(data.splitlines(),1):
 try:r=json.loads(line)
 except (TypeError,ValueError):bad+=1;continue
 ev=r.get('event','missing');pid=r.get('pid');counts[ev]+=1;reported_drops+=int(r.get('dropped_since_last',0) or 0);event_losses+=int(bool(r.get('dropped_since_last')))
 if ev=='trace_cap':caps.append({'pid':pid,'max_bytes':r.get('max_bytes')})
 if r.get('record_cap_reached'):seqcap.append(pid)
 if ev in ('load_data','load_file') and r.get('rc')==0:
  h=r.get('handle');key=(pid,h)
  for fkey in [k for k,v in funcs.items() if k[0]==pid and v['bin']==h]:funcs.pop(fkey)
  if key in active:found[active.pop(key)]['end']='same_handle_new_load_observed'
  sha=r.get('pre_call_sha256')
  if h not in (None,'0x0','(nil)') and sha in known:
   active[key]=len(found);found.append({'line':i,'pid':pid,'handle':h,'sha256':sha,'hash_status':r.get('hash_status'),'load_event':ev,'load_monotonic_ns':r.get('monotonic_ns'),'entry_count':0,'named_count':0,'launch_begin_count':0,'launch_end_count':0,'function_handles':[],'end':'end_of_snapshot'})
 if ev=='unload' and r.get('rc')==0:
  h=r.get('bin');key=(pid,h)
  if key in active:found[active.pop(key)]['end']='unload_observed'
  for fkey in [k for k,v in funcs.items() if k[0]==pid and v['bin']==h]:funcs.pop(fkey)
 if ev in ('get_function','get_function_entry') and r.get('rc')==0:
  func=r.get('func');bin=r.get('bin');key=(pid,bin); fk=(pid,func)
  if func not in (None,'0x0','(nil)'):
   funcs.pop(fk,None)
   if key in active:
    k=active[key]; found[k]['entry_count' if ev=='get_function_entry' else 'named_count']+=1
    item={'func':func,'api':ev,'entry':r.get('entry'),'line':i,'monotonic_ns':r.get('monotonic_ns')}
    if ev=='get_function':
     try:
      name=bytes.fromhex(r.get('name_hex','')).decode('utf-8')
      if 'dispatchffncombine' in name.lower():item['target_name']=name
     except (TypeError,ValueError,UnicodeDecodeError):pass
    found[k]['function_handles'].append(item);funcs[fk]={'bin':bin,'load_index':k}
 if ev in ('launch_begin','launch_host_args'):
  v=funcs.get((pid,r.get('func')))
  if v is not None and active.get((pid,v['bin']))==v['load_index']:
   k=v['load_index'];found[k]['launch_begin_count' if ev=='launch_begin' else 'launch_end_count']+=1
   if ev=='launch_host_args' and r.get('rc')!=0:coverage['candidate_launch_failed_rc']+=1
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'trace_bytes':len(data),'trace_sha256':hashlib.sha256(data).hexdigest(),'trace_mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'partial_final_line':snapshot_incomplete,'bad_lines':bad,'trace_caps':caps,'record_caps':seqcap,'event_counts':dict(counts),'reported_dropped_lower_bound':reported_drops,'events_reporting_dropped':event_losses,'matched_loads':found,'matched_load_count':len(found),'extra_coverage':dict(coverage),'limitations':['pre_call_sha256 only hashes pre-call memory/file candidate; runtime loading exact bytes unproved','same PID/handle association within observed records is a candidate, not a verified handle lifetime; dropped records, nested/multithread calls, reused handles and truncation can hide transitions','load/entry/launch observed before cap only; no missing-event exclusion; task/profiler graph identity is not established']}
print(json.dumps(result,sort_keys=True))
'''
cmd=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code]
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(cmd)],capture_output=True,text=True,timeout=45)
print(r.stdout,end='');print(r.stderr,end='');raise SystemExit(r.returncode)
