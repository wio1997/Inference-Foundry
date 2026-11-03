"""One-shot CPU review waits on the exact owned acquisition handle."""
import json,os,signal,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168')
APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/joint168'
state={};child=None
def save():
 p=ROOT/'POST-STATUS.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'POST-STATUS.json')
def run(name,limit=2700):
 global child
 log=(ROOT/('post-'+name+'.log')).open('w')
 child=subprocess.Popen(['docker','exec','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','wio-vllm-v41-repro171-20261102','python3',APP+'/'+name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(state='CPU_REVIEW_RUNNING',analysis_script=name,host_child_pid=child.pid,host_child_start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]);save()
 try:rc=child.wait(timeout=limit)
 except subprocess.TimeoutExpired:
  # Kill only this exact CPU-only analysis command,never the inference service.
  subprocess.run(['docker','exec','wio-vllm-v41-repro171-20261102','pkill','-TERM','-f','^python3 '+APP+'/'+name+'$'],check=False);rc=child.wait(timeout=30)
  raise TimeoutError(name+'CPU review deadline,returncode'+str(rc))
 child=None;state.setdefault('analyses',[]).append({'script':name,'returncode':rc});save()
 return rc
if __name__=='__main__':
 fd=os.open(ROOT/'POST-STATUS.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
 state={'state':'WAITING_EXACT_ACQUISITION_HANDLE','pid':os.getpid(),'start_ticks':Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')',1)[1].split()[19],'started_unix':time.time(),'CPU_only':True,'does_not_restart_or_reload_service':True,'overall_complete':False};save();end=time.monotonic()+6000
 try:
  while True:
   a=json.loads((ROOT/'STATUS.json').read_text());h=json.loads((ROOT/'SUBMITTED.json').read_text())
   if a['state'] in ['ORIGINAL_SERVICE_RESTORED_REVIEW_PENDING','ACQUISITION_FAILED_ORIGINAL_SERVICE_RESTORED']:
    state['acquisition_terminal_state']=a['state'];save();break
   p=Path('/proc/'+str(h['pid'])+'/stat');live=p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==h['start_ticks']
   if not live:raise RuntimeError('Acquisitionhandle missing without terminal restore receipt;no automatic model action')
   if time.monotonic()>end:raise TimeoutError('Deferred CPUreview wait deadline;no model action')
   state.update(acquisition_state=a['state'],last_verified_handle=h,last_verified_live_unix=time.time());save();time.sleep(5)
  # Reviews are independent;an error in one does not discard the other evidence.
  histories=run('analyze_histories.py');export=run('offline_joint.py')
  state.update(state='CPU_REVIEWS_RETURNED' if histories==export==0 else 'CPU_REVIEWS_RETURNED_WITH_ERRORS',finished_unix=time.time());save()
 except BaseException:
  state.update(state='CPU_REVIEW_ERROR',error=traceback.format_exc(),finished_unix=time.time());save();raise
