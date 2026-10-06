"""One controller-owned bounded native observation; original service unchanged."""
import hashlib,json,os,select,sys,time
from pathlib import Path
started=time.monotonic_ns();ROOT=Path(sys.argv[1]);RESEARCH=ROOT.parents[1]/'research/decode_path_20261006/native_boundary'
sys.path.insert(0,str(RESEARCH));from perf_boundary_B import Observer,function_offsets
plan=json.loads((ROOT/'observer_plan.json').read_text());selfcheck=json.loads((RESEARCH/'CPU_result_B.json').read_text());assert selfcheck['passed'] and selfcheck['model_requests']==0
assert hashlib.sha256((RESEARCH/'fixture.so').read_bytes()).hexdigest()==selfcheck['probe_source_sha256']
pids=[x['pid'] for x in plan['workers']];before={};maps={}
for row in plan['workers']:
 pid=row['pid'];parts=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split();assert parts[19]==row['start_ticks'];assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot_id']
 before[str(pid)]=dict(start_ticks=parts[19],sched_runtime_ns=int(Path('/proc/%d/schedstat'%pid).read_text().split()[0]));maps[str(pid)]=Path('/proc/%d/maps'%pid).read_text()
probes={};identity=[]
for label in ['native','workspace']:
 cfg=plan[label];paths=[Path('/proc/%d/root'%pid+cfg['path']) for pid in pids]
 assert len({(p.stat().st_dev,p.stat().st_ino) for p in paths})==1
 assert all(hashlib.sha256(p.read_bytes()).hexdigest()==cfg['sha256'] for p in paths)
 path=str(paths[-1])
 if label=='native':offsets={name:dict(va=va,offset=va-cfg['text_va_minus_file_offset']) for name,va in cfg['vas'].items()}
 else:
  offsets,sha=function_offsets(path,list(cfg['symbols'].values()));assert sha==cfg['sha256'];offsets={name:offsets[symbol] for name,symbol in cfg['symbols'].items()}
 identity.append(dict(kind=label,path=path,sha256=cfg['sha256'],offsets=offsets))
 for name,info in offsets.items():
  for side in ['enter','exit']:probes[name+'_'+side]=dict(path=path,offset=info['offset'],**{'return':side=='exit'})
(ROOT/'probe_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
prior_groups=Path('/sys/kernel/tracing/uprobe_events').read_text();obs=Observer(probes,pids,RESEARCH/'fixture.so','glm53_mc2_boundary');registered=time.monotonic_ns();active=False;done=None
try:
 print(json.dumps(dict(ready=True,pid=os.getpid(),worker_pids=pids,event_group=obs.group,probes=list(probes),NPU_profile=False)),flush=True)
 assert sys.stdin.readline().strip()=='go';ready=time.monotonic_ns();cpu_start=time.process_time_ns();obs.enable();enabled=time.monotonic_ns();active=True
 print(json.dumps(dict(go=True,start_ns=ready,enabled_ns=enabled,boundary_task_clock=True)),flush=True)
 deadline=time.monotonic()+90
 while True:
  r,_,_=select.select([sys.stdin],[],[],0.01);obs.drain()
  if r:assert sys.stdin.readline().strip()=='done';break
  assert time.monotonic()<deadline,'observer timeout'
 obs.disable();active=False;end=time.monotonic_ns();cpu_end=time.process_time_ns();events=sorted(obs.events,key=lambda x:x.get('wall_ns',0));lost=sum(x.get('lost',0) for x in events)
 raw=dict(events=events,start_ns=ready,enabled_ns=enabled,end_ns=end,before=before,maps=maps,probes=probes,observer_cpu_ns=cpu_end-cpu_start,group=obs.group,lost=lost,clock='CLOCK_MONOTONIC wall + same-TID PERF_COUNT_SW_TASK_CLOCK sample group',limits=['Producer only; no consumer runnable/idle inference.','Probe overhead and kernel work included in task CPU; diagnostic wall is not performance comparison.','Last workspace-to-return phase includes submission and cleanup; actual queue push not observed.'])
 payload=json.dumps(raw,separators=(',',':')).encode()+b'\n';(ROOT/'native_boundary_raw.json').write_bytes(payload)
 done=dict(done=True,boundary_task_clock=True,events=len(events),lost=lost,ready_ns=ready,started_ns=started,registration_end_ns=registered,enabled_ns=enabled,go_received_ns=ready,raw_sha256=hashlib.sha256(payload).hexdigest(),raw_bytes=len(payload),window_ns=end-ready,observer_cpu_ns=cpu_end-cpu_start,performance_claim=False)
 (ROOT/'observer_final.json').write_text(json.dumps(done,indent=2)+'\n')
finally:
 if active:obs.disable()
 obs.close();remaining=Path('/sys/kernel/tracing/uprobe_events').read_text();assert obs.group+'/' not in remaining
 (ROOT/'probe_cleanup.json').write_text(json.dumps(dict(group=obs.group,removed=True,other_groups_unchanged=remaining==prior_groups))+'\n')
if done:print(json.dumps(done),flush=True)
