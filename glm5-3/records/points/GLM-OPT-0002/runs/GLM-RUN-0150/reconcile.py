from pathlib import Path
import json,sys,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import guard,start_watchdog
start_watchdog();r=Path(__file__).parent
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:"+str(r)+":$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(r/"client.py")+" reconcile"
guard();z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=360);(r/"native_reconcile.stdout").write_bytes(z.stdout);(r/"native_reconcile.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"final"],capture_output=True,timeout=240);(r/"final.stdout").write_bytes(z.stdout);(r/"final.stderr").write_bytes(z.stderr);z.check_returncode()
atomic_json(r/"reconcile_summary.json",dict(at=utc(),valid=True,SDKinit_finalize0=True,client=json.loads((r/"client_reconcile_summary.json").read_text()),models_operations=0,frontend_operations=0,original149_not_replayed=True,same149_native32=True))
