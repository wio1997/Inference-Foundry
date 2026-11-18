"""Bounded two-slot CPU-only full-rank SDK work; original captures are immutable."""
from pathlib import Path
import concurrent.futures,datetime,json,os,queue,subprocess,threading,time
CPU=Path('/PRIVATE_TASK_ROOT/native_profile_CPU_g838');OUT=CPU/'remaining_D_g866';OUT.mkdir(exist_ok=False);began=time.monotonic();lock=threading.Lock();failed=threading.Event();jobs={};results={};pending=queue.Queue()
parity=json.loads((CPU/'D0_native_backend_parity_g861.json').read_text());assert parity['full_D0_independent_CPU_processing_parity_accepted']
for rank in range(1,8):assert not (CPU/f'jobs_cli_g853/rank{rank}').exists();pending.put(rank)
def state(phase):
 with lock:
  value={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-began,'active':dict(jobs),'results':dict(results),'remaining_ranks':list(pending.queue),'CPU_slots':['160-171','172-183'],'NPU_devices':[],'max_workers':2,'scope':'OfflinefullrawDBparse only; noNPUrecapture; nocontainer/model/servicechanges'};p=OUT/'state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def slot(cpus):
 while not failed.is_set():
  try:rank=pending.get_nowait()
  except queue.Empty:return
  argv=['python3','-u',str(CPU/'adapter_g853/parse_D_rank.py'),'--rank',str(rank),'--cpu-set',cpus]
  with (OUT/f'rank{rank}.log').open('x') as stream:p=subprocess.Popen(argv,cwd=CPU,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
  with lock:jobs[str(rank)]={'pid':p.pid,'argv':argv,'CPU_slot':cpus,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  state('bounded_two_slot_CPU_DB_parsing');rc=p.wait()
  with lock:results[str(rank)]={'returncode':rc,'result':str(CPU/f'jobs_cli_g853/rank{rank}/result.json')};jobs.pop(str(rank))
  if rc:failed.set()
  state('failure_no_new_rank_launches' if failed.is_set() else 'bounded_two_slot_CPU_DB_parsing')
state('starting_independent_CPU_ranks')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(slot,cpus) for cpus in ['160-171','172-183']]
 for f in futures:f.result()
if failed.is_set():state('failed_requires_parent_review');raise RuntimeError('At least one fullrank CPU parse failed; preserveallcopies, noautomaticretry')
assert set(results)=={str(n) for n in range(1,8)} and all(v['returncode']==0 for v in results.values());state('all7_CPU_SDK_complete_pending_parent_device_review')
