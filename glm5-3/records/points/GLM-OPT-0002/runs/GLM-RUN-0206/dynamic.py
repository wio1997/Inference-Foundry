from pathlib import Path
import json,sys,subprocess,shlex,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from owner_guard import start_watchdog,guard
from phase_runner import atomic_json,utc
from native_client_execution import run_native_client
import controls
start_watchdog();r=Path(__file__).parent
for label in ["dynamic_initial"]:
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),label],capture_output=True,timeout=240);(r/(label+".stdout")).write_bytes(z.stdout);(r/(label+".stderr")).write_bytes(z.stderr);z.check_returncode()
controls.verify("dynamic_policy_before")
stats=r.parent/"GLM-RUN-0204/token_memo_stats.json"
atomic_json(r/"dynamic_memo_before.json",json.loads(stats.read_text()))
native=["python3","-u","/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py","script",str(r/"arrivals_client.py")]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; exec "+shlex.join(native)
exe=run_native_client(["docker","exec","glm52-single","bash","-c",shell],record_path=r/"native_dynamic_execution.json",stdout_path=r/"native_dynamic.stdout",stderr_path=r/"native_dynamic.stderr",expected_native_argv=native,timeout_s=1500,guard=guard)
acks=[json.loads(l)for l in(r/"native_dynamic.stdout").read_text().splitlines()if l.startswith('{"event":')]
assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
assert exe["native_client"]and not exe["native_client_alive"]and not exe["signal_attempts"]and exe["status"]=="succeeded"
s=json.loads((r/"dynamic_summary.json").read_text());assert s["functional_acceptance"]and s["actual_new_requests"]==12and s["effective_public_output_tokens"]==10112
controls.verify("dynamic_policy_after")
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_final"],capture_output=True,timeout=240);(r/"epoch_final.stdout").write_bytes(z.stdout);(r/"epoch_final.stderr").write_bytes(z.stderr);z.check_returncode()
atomic_json(r/"dynamic_memo_after.json",json.loads(stats.read_text()))
atomic_json(r/"dynamic_wrapper_summary.json",dict(at=utc(),functional_acceptance=True,new_native_outputs=10112,native_complete_requests=12,actualSDK_init_finalize=[acks[0],acks[-1]],policies_unchanged=True,private_policy_writes=0,experimental_cadence=1,native_dense_Graph_205=True,models_operations=0,frontend_operations=0,capacity_verdict="pending raw audit"))
print(json.dumps(dict(outputs=s["effective_public_output_tokens"],elapsed_s=s["elapsed_s"],SLO=s["diagnostic_reference_SLO"])))
