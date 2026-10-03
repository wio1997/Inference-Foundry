from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from owner_guard import start_watchdog,guard
from phase_runner import atomic_json,utc
start_watchdog();r=Path(__file__).parent
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"dynamic_initial"],capture_output=True,timeout=240);(r/"dynamic_initial.stdout").write_bytes(z.stdout);(r/"dynamic_initial.stderr").write_bytes(z.stderr);z.check_returncode()
shell="export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(r/"arrivals.py")
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=1500)
(r/"native_dynamic.stdout").write_bytes(z.stdout);(r/"native_dynamic.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(x)for x in z.stdout.decode().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
s=json.loads((r/"dynamic_summary.json").read_text());assert s["functional_acceptance"]and s["actual_new_requests"]==4 and s["effective_public_output_tokens"]==16384
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=s);atomic_json(r/"manifest.json",m);print(json.dumps(dict(outputs=s["effective_public_output_tokens"],elapsed_s=s["elapsed_s"],SLO=s["diagnostic_reference_SLO"])))

z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_final"],capture_output=True,timeout=240);(r/"epoch_final.stdout").write_bytes(z.stdout);(r/"epoch_final.stderr").write_bytes(z.stderr);z.check_returncode()
