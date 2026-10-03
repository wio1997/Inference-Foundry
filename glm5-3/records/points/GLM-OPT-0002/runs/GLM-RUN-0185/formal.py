from pathlib import Path
import json,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from native_client_execution import run_native_client
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;window=sys.argv[1]
native=["python3","-u","/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py","script",str(r/"formal_client.py"),str(r/window),r.name,"--cache_mode","resident_observed","--deadline_s","6600"]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; exec "+shlex.join(native)
execution=run_native_client(["docker","exec","glm52-single","bash","-c",shell],record_path=r/(window+"_native_client_execution.json"),stdout_path=r/(window+"_native_formal.stdout"),stderr_path=r/(window+"_native_formal.stderr"),expected_native_argv=native,timeout_s=7200,guard=guard,grace_s=120)
acks=[json.loads(x)for x in(r/(window+"_native_formal.stdout")).read_text().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
assert not execution["native_client_alive"]and execution["native_client"]is not None and not execution["timed_out"]
s=json.loads((r/window/"formal_summary.json").read_text());assert s["valid"]and s["effective_output_tokens"]==245760
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_"+window+"_final"],capture_output=True,timeout=240);(r/(window+"_epoch_final.stdout")).write_bytes(z.stdout);(r/(window+"_epoch_final.stderr")).write_bytes(z.stderr);z.check_returncode()
out=dict(at=utc(),functional_acceptance=True,formal=s,actual_client_SDK_init=acks[0],actual_client_SDK_finalize=acks[-1],native_client_execution=execution,model_operations=0,gateway_operations=0,terminal_raw_audit_pending=True,limits=s["limits"])
atomic_json(r/(window+"_wrapper_summary.json"),out);print(json.dumps(out))
