"""Hold existing canonical DS P/D workload flock for one bounded external job."""
from pathlib import Path
import datetime,fcntl,json,os,socket,time
R=Path('/PRIVATE_TASK_ROOT/sg_pd_g832')
assert socket.gethostname()=='<ROLE_HOSTNAME>'
lock=Path('/root/.codex/coord/deepseek-v41-p3d4-workload.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
o=R/'workload_guard_g929';o.mkdir(exist_ok=False)
began=time.monotonic();bound=43200
def write(phase):
 d={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-began,'bound_s':bound,'job':'SG_P832_D918_paged924_quality_matrix_profile_G929','pair':'P2203/D2204','lock_scope':'DS workload only, no GLM lock','P_container_id':'0322fbcc972c32f1f805397cd33b9c2549cb8d7bcbad6545d6bb0de42027154a','D_container_id':json.loads((R/'pair_identity_g929.json').read_text())['D_container_id']}
 p=o/'state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
write('holding')
while time.monotonic()-began<bound and not (o/'stop.request').exists():
 s=json.loads((R/'state.json').read_text());assert s['generation']==832 and s['container_id']=='0322fbcc972c32f1f805397cd33b9c2549cb8d7bcbad6545d6bb0de42027154a'
 assert str(R/'node_controller.py').encode() in Path('/proc',str(s['pid']),'cmdline').read_bytes()
 time.sleep(15)
write('released_after_stop_request' if (o/'stop.request').exists() else 'expired_requires_external_job_audit')
