"""Hold existing canonical DS P/D workload flock for one bounded external job."""
from pathlib import Path
import datetime,fcntl,json,os,socket,time
R=Path('/PRIVATE_TASK_ROOT/vllm_pd_g727')
assert socket.gethostname()=='<ROLE_HOSTNAME>'
lock=Path('/root/.codex/coord/deepseek-v41-p3d4-workload.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
o=R/'profile_guard_g790';o.mkdir(exist_ok=False)
began=time.monotonic();bound=14400
def write(phase):
 d={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-began,'bound_s':bound,'job':'native727_profile_warm_capture_G790','pair':'P2203/D2204','lock_scope':'DS workload only, no GLM lock','P_container_id':'d30b701d751751b522af6e472b982eabb198550234c086871176e368170c7b70','D_container_id':'99def3aa5173e209e3bf65c8d97868785e5dbcfc068baf0ba48cdc1c3680a5e9'}
 p=o/'state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
write('holding')
while time.monotonic()-began<bound and not (o/'stop.request').exists():
 s=json.loads((R/'state.json').read_text());assert s['generation']==727 and s['container_id']=='d30b701d751751b522af6e472b982eabb198550234c086871176e368170c7b70'
 assert str(R/'node_controller.py').encode() in Path('/proc',str(s['pid']),'cmdline').read_bytes()
 time.sleep(15)
write('released_after_stop_request' if (o/'stop.request').exists() else 'expired_requires_external_job_audit')
