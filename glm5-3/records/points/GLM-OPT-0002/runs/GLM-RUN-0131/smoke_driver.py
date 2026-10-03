from pathlib import Path
import sys,subprocess,shlex,json
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent
for name in["public_smoke"]:
 guard();shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:"+str(r)+":$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_run131/native_acl_lifecycle.py script "+shlex.quote(str(r/(name+".py")))
 z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=600);(r/("native_"+name+".stdout")).write_bytes(z.stdout);(r/("native_"+name+".stderr")).write_bytes(z.stderr);z.check_returncode()
 acks=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')]
 assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"pilot_final"],capture_output=True,timeout=240);(r/"pilot_final.stdout").write_bytes(z.stdout);(r/"pilot_final.stderr").write_bytes(z.stderr);z.check_returncode()
atomic_json(r/"public_smoke_SDK_summary.json",dict(at=utc(),SDK0=True,smoke=json.loads((r/"public_smoke_summary.json").read_text())))
