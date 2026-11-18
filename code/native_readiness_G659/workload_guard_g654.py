"""Hold existing canonical DS P/D workload flock for one bounded external job."""
from pathlib import Path
import datetime,fcntl,json,os,socket,time
R=Path('/PRIVATE_TASK_ROOT/vllm_pd_g630')
assert socket.gethostname()=='<ROLE_HOSTNAME>'
lock=Path('/root/.codex/coord/deepseek-v41-p3d4-workload.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
o=R/'quality_guard_g654';o.mkdir(exist_ok=False)
began=time.monotonic();bound=10800
def write(phase):
 d={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-began,'bound_s':bound,'job':'native630_quality_G655','pair':'P2203/D2204','lock_scope':'DS workload only, no GLM lock','P_container_id':'36d18369f08e7ef174ff294578f714f65dc0b4a2824fd4ff7c0778b8d2292d40','D_container_id':'6b8de99e90f4fa483d6077926976c20a27c8fdf31a0da836c61c7b587d85f2f9'}
 p=o/'state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
write('holding')
while time.monotonic()-began<bound and not (o/'stop.request').exists():
 s=json.loads((R/'state.json').read_text());assert s['generation']==630 and s['container_id']=='36d18369f08e7ef174ff294578f714f65dc0b4a2824fd4ff7c0778b8d2292d40'
 assert str(R/'node_controller.py').encode() in Path('/proc',str(s['pid']),'cmdline').read_bytes()
 time.sleep(15)
write('released_after_stop_request' if (o/'stop.request').exists() else 'expired_requires_external_job_audit')
