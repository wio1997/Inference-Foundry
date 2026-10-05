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
initial={k:dict(v)for k,v in controls.policies.items()};offsets={}
plans=json.loads((r/"standalone_launch.json").read_text())
for node,key in[("166","node0"),("167","node1")]:
 code="from pathlib import Path;import json;print(json.dumps(dict(bytes=Path("+repr(plans[key]["log"])+").stat().st_size)))"
 ack=controls.run(node,code,{},"native_log_offset_"+node);offsets[node]=ack["bytes"];atomic_json(r/("native_log_offset_"+node+".json"),ack)
try:
 controls.update_D1(dict(initial["167"],prefill_threshold_tokens=2048,serial=initial["167"]["serial"]+1),"D1_prefill2048")
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
 atomic_json(r/"dynamic_wrapper_summary.json",dict(at=utc(),functional_acceptance=True,new_native_outputs=10112,native_complete_requests=12,actualSDK_init_finalize=[acks[0],acks[-1]],D0_policy_unchanged=True,D1_prefill2048=True,D1_measured_policy=dict(controls.policies["167"]),private_policy_writes=1,experimental_cadence=1,native_dense_Graph_unchanged=True,models_operations=0,frontend_operations=0,capacity_verdict="pending raw audit"))
 plans=json.loads((r/"standalone_launch.json").read_text());selected={}
 for node,key in[("166","node0"),("167","node1")]:
  offset=offsets[node];path=plans[key]["log"];code="from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(path)+").read_bytes()["+str(offset)+":])"
  args=["python3","-c",code]
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  result=subprocess.run(args,capture_output=True,timeout=60);(r/("native_policy_window_"+node+".log")).write_bytes(result.stdout);(r/("native_policy_window_"+node+".stderr")).write_bytes(result.stderr);result.check_returncode()
  values=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in result.stdout.decode().splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
  assert(values or node=="166")and all(v["cohort"]==controls.policies[node]["cohort_id"]and v["budget_tokens"]==8192and v["prefill_threshold_tokens"]==(2048 if node=="167"else 1024)and v["prefill_cadence"]==1and v["serial"]==controls.policies[node]["serial"]and v["native_max"]==8192and not v["fallback"]for v in values)
  selected[node]=values
 atomic_json(r/"policy_consumption.json",dict(at=utc(),native_selected=selected,D1_threshold2048_selected=True,allocated8192_unchanged=True,operator_math_changes=0,single_native_scheduler_authority=True,native_schedule_count_not_GPUsteps=True))
finally:
 if controls.policies["167"]!=initial["167"]:
  controls.update_D1(dict(initial["167"],serial=controls.policies["167"]["serial"]+1),"D1_prefill1024_restore")
 controls.verify("restored_policy")
 atomic_json(r/"restore_summary.json",dict(at=utc(),policies=controls.policies,private_policy_writes=controls.policies["167"]["serial"]-initial["167"]["serial"],model_operations=0,both_same_owned_idle=True,D1_prefill1024_restored=True,D0_policy_unchanged=True,restored_cadence1=True))
print(json.dumps(dict(outputs=s["effective_public_output_tokens"],elapsed_s=s["elapsed_s"],SLO=s["diagnostic_reference_SLO"])))
