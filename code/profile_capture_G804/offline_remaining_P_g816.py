from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime,fcntl,json,os,socket,subprocess,time
assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT');r=root/'vllm_pd_g727';o=r/'offline_remaining_g816';o.mkdir(exist_ok=False);lock=(o/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pilot=json.loads((r/'offline_rank0_g809/result.json').read_text());assert pilot['returncode']==0 and pilot['original_raw_SHA256_unchanged'] and pilot['trace_view_present'] and pilot['op_statistic_present']
assert (Path(pilot['output'])/'kernel_details.csv').stat().st_size>0
assert json.loads((r/'raw_logical_work_g812/summary.json').read_text())['all8_normalized_scheduler_sequences_identical']
def save(path,value):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(path)
save(o/'state.json',{'phase':'offline_CPU_batch_running','pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ranks':list(range(1,8)),'workers':2,'CPU_per_worker':8,'SDK_inner_deadline_per_rank_s':600,'allrank_profile_accepted':False,'scope':'P0SDKcompleted; pipeline remaining7fulloriginalcaptureranks onisolatedcopies. Actualdevice/logicalacceptance remainsparentreview. NoNPUload or servicechange.'})
def worker(slot):
 cpu=['144-151','152-159'][slot];results=[]
 for rank in range(slot+1,8,2):
  cmd=['python3','-u',str(r/'offline_rank_worker_g816.py'),'--rank',str(rank),'--cpu-set',cpu]
  with (o/f'rank{rank}_parent.log').open('x') as stream:
   p=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);h={'pid':p.pid,'argv':cmd,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank':rank,'slot':slot};save(o/f'rank{rank}_actual_host_handle.json',h);code=p.wait(timeout=1100)
  result={'rank':rank,'returncode':code,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()};save(o/f'rank{rank}_parent_result.json',result);results.append(result)
  if code:break
 return results
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,range(2)))
flat=[item for group in results for item in group];passed=len(flat)==7 and all(item['returncode']==0 for item in flat);save(o/'state.json',{'phase':'SDK_batch_complete_pending_allrank_review' if passed else 'failed_requires_review','pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':flat,'allrank_profile_accepted':False});print(json.dumps({'batch_completed':passed,'ranks':flat}),flush=True)
