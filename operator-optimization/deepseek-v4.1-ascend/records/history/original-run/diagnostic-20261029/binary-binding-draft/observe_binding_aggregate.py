import subprocess,shlex
code=r'''import os,json,collections,datetime
p='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic/runtime/metadata/binding-trace.jsonl'
with open(p,'rb') as f:
 st=os.fstat(f.fileno()); data=f.read(min(st.st_size,16777216))
counts=collections.Counter();bad=0; drops=0; dropped_records=0; cap=[]; rec_cap=[]; active={}; prior=collections.Counter(); targets=[]; incomplete=data and not data.endswith(b'\n')
for line in data.splitlines():
 try: r=json.loads(line)
 except (ValueError,TypeError): bad+=1;continue
 ev=r.get('event','missing'); counts[ev]+=1; pid=r.get('pid'); drops+=int(r.get('dropped_since_last',0) or 0);dropped_records+=int(bool(r.get('dropped_since_last',0))); 
 if ev=='trace_cap': cap.append({'pid':pid,'max_bytes':r.get('max_bytes')})
 if r.get('record_cap_reached'):rec_cap.append(pid)
 if ev in ('load_data','load_file') and r.get('rc')==0:
  h=r.get('handle'); key=(pid,h)
  if h and h not in ('(nil)','0x0'):
   prior[key]+=1
   if key in active: active[key].append(r)
   else: active[key]=[r]
 if ev=='unload' and r.get('rc')==0:
  active.pop((pid,r.get('bin')),None)
 if ev=='get_function' and r.get('name_status')=='ok':
  try: name=bytes.fromhex(r.get('name_hex','')).decode('utf-8')
  except (ValueError,UnicodeDecodeError):continue
  if 'dispatchffncombine' in name.lower():
   h=r.get('bin'); candidates=active.get((pid,h),[])
   status=('invalid_get_function' if r.get('rc')!=0 else 'invalid_handle' if not h or h in ('(nil)','0x0') else 'missing_observed_load' if not candidates else 'unique_observed_load' if len(candidates)==1 else 'ambiguous_multiple_loads')
   info={'pid':pid,'name':name,'get_rc':r.get('rc'),'bin':h,'func':r.get('func'),'get_monotonic_ns':r.get('monotonic_ns'),'binding_status':status,'same_handle_observed_loads_in_lifetime':len(candidates),'prior_observed_loads_same_handle':prior[(pid,h)]}
   if len(candidates)==1: 
    q=candidates[0]; info['load']={'event':q.get('event'),'rc':q.get('rc'),'pre_call_sha256':q.get('pre_call_sha256'),'hash_status':q.get('hash_status'),'monotonic_ns':q.get('monotonic_ns')}
   targets.append(info)
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_bytes':len(data),'max_bytes':16777216,'mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'partial_final_line_at_snapshot':bool(incomplete),'bad_lines':bad,'event_counts':dict(counts),'reported_dropped_total_lower_bound':drops,'records_reporting_dropped':dropped_records,'trace_cap_markers':cap,'record_cap_reached_pids':rec_cap,'target_get_function_count':len(targets),'non_target_event_count':sum(counts.values())-len(targets),'target_get_function_records':targets,'limitations':['event records can be dropped under lock contention, truncate at caps, or race event completion across threads; absence of observed unload or load does not prove lifetime','pre_call_sha256 is a candidate digest, not proof of exact opened/executed bytes','same pid and observed handle association alone does not establish profiler task or generated request identity']},ensure_ascii=False,sort_keys=True))
'''
cmd=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code]
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(cmd)],capture_output=True,text=True,timeout=45)
print(r.stdout,end='');print(r.stderr,end='');raise SystemExit(r.returncode)
