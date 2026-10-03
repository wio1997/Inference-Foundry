from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(r/"formal_client.py")+" "+str(r/sys.argv[1])+" "+r.name+" --cache_mode resident_observed"
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=1700);(r/(sys.argv[1]+"_native_formal.stdout")).write_bytes(z.stdout);(r/(sys.argv[1]+"_native_formal.stderr")).write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(x)for x in z.stdout.decode().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
s=json.loads((r/sys.argv[1]/"formal_summary.json").read_text());assert s["valid"]and s["effective_output_tokens"]==16384
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_"+sys.argv[1]+"_final"],capture_output=True,timeout=240);(r/(sys.argv[1]+"_epoch_final.stdout")).write_bytes(z.stdout);(r/(sys.argv[1]+"_epoch_final.stderr")).write_bytes(z.stderr);z.check_returncode()
out=dict(at=utc(),functional_acceptance=True,formal=s,actual_client_SDK_init=acks[0],actual_client_SDK_finalize=acks[-1],model_operations=0,gateway_operations=0,terminal_raw_audit_pending=True,limits=s["limits"])
atomic_json(r/(sys.argv[1]+"_wrapper_summary.json"),out);print(json.dumps(out))
