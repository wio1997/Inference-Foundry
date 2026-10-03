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
initial={k:dict(v)for k,v in controls.policies.items()}
for node in ["166","167"]:
 plugin="local_pp190"if node=="166"else"local_pp184"
 code="from pathlib import Path;import json;print(json.dumps(dict(bytes=Path('/data/tiankuan/wio/glm52-pd/deploy/logs/"+plugin+"_"+node+".log').stat().st_size)))"
 ack=controls.run(node,code,{},"native_log_offset_"+node);atomic_json(r/("native_log_offset_"+node+".json"),ack)
try:
 controls.update_D0(dict(initial["166"],prefill_cadence=2,serial=2),"D0_cadence2")
 controls.verify("dynamic_c2_policy_before")
 stats=r.parent/"GLM-RUN-0190/token_memo_stats.json"
 atomic_json(r/"dynamic_memo_before.json",json.loads(stats.read_text()))
 native=["python3","-u","/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py","script",str(r/"arrivals_client.py")]
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; exec "+shlex.join(native)
 exe=run_native_client(["docker","exec","glm52-single","bash","-c",shell],record_path=r/"native_dynamic_execution.json",stdout_path=r/"native_dynamic.stdout",stderr_path=r/"native_dynamic.stderr",expected_native_argv=native,timeout_s=1500,guard=guard)
 acks=[json.loads(l)for l in(r/"native_dynamic.stdout").read_text().splitlines()if l.startswith('{"event":')]
 assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 assert exe["native_client"]and not exe["native_client_alive"]and not exe["signal_attempts"]and exe["status"]=="succeeded"
 selected={}
 for node in ["166","167"]:
  offset=json.loads((r/("native_log_offset_"+node+".json")).read_text())["bytes"]
  plugin="local_pp190"if node=="166"else"local_pp184"
  code="from pathlib import Path;import sys;sys.stdout.buffer.write(Path('/data/tiankuan/wio/glm52-pd/deploy/logs/"+plugin+"_"+node+".log').read_bytes()["+str(offset)+":])"
  args=["python3","-c",code]
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,capture_output=True,timeout=60);(r/("native_policy_window_"+node+".log")).write_bytes(z.stdout);(r/("native_policy_window_"+node+".stderr")).write_bytes(z.stderr);z.check_returncode()
  values=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in z.stdout.decode().splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
  assert (values or node=="167") and all(v["prefill_cadence"]==(2 if node=="166"else 1)and v["budget_tokens"]==8192and v["prefill_threshold_tokens"]==1024and v["serial"]==controls.policies[node]["serial"]and not v["fallback"]for v in values);selected[node]=values
 applied=[json.loads(l.split("GLM_LOCAL_PREFILL_CADENCE_APPLIED ",1)[1])for l in(r/"native_policy_window_166.log").read_text().splitlines()if"GLM_LOCAL_PREFILL_CADENCE_APPLIED "in l]
 assert all(v["serial"]==2 and v["prefill_cadence"]==2 and v["local_throttle"]and not v["incoming_native_DP_flag"]and v["scheduled_prefill_tokens"]==0and v["has_native_decode"]and not v["native_capacity_bound"]and v["ready_decode_candidates_before"]>0 for v in applied)
 bypass=[json.loads(l.split("GLM_LOCAL_PREFILL_CADENCE_BYPASSED ",1)[1])for l in(r/"native_policy_window_166.log").read_text().splitlines()if"GLM_LOCAL_PREFILL_CADENCE_BYPASSED "in l]
 assert all(v["serial"]==2and v["prefill_cadence"]==2and v["reason"]=="no_native_decode_candidate_tokens"and not v["local_throttle"]and v["has_native_decode"]for v in bypass)
 priorD1=json.loads((r.parent/"GLM-RUN-0189/policy_consumption.json").read_text())["native_selected"]["167"]
 assert priorD1 and all(v["prefill_cadence"]==1and v["serial"]==3and not v["fallback"]for v in priorD1)
 atomic_json(r/"policy_consumption.json",dict(at=utc(),native_selected=selected,D0_actual_APPLIED=applied,D0_actual_BYPASSED=bypass,positive_native_scheduled_ready_decode=any(v["scheduled_ready_decode_tokens"]>0for v in applied),positive_native_scheduled_total=any(v["scheduled_total_tokens"]>0for v in applied),D1_same_epoch_prior_SELECTED=priorD1,D1_cadence1_unchanged=True,allocated8192_unchanged=True,operator_math_changes=0,single_native_scheduler_authority=True,sequence_is_not_GPUsteps=True,bounded_first4_per_marker=True))
 s=json.loads((r/"dynamic_summary.json").read_text());assert s["functional_acceptance"]and s["actual_new_requests"]==12and s["effective_public_output_tokens"]==10112
 controls.verify("dynamic_c2_policy_after")
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_final"],capture_output=True,timeout=240);(r/"epoch_final.stdout").write_bytes(z.stdout);(r/"epoch_final.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"dynamic_memo_after.json",json.loads(stats.read_text()))
 atomic_json(r/"dynamic_wrapper_summary.json",dict(at=utc(),functional_acceptance=True,new_native_outputs=10112,native_complete_requests=12,actualSDK_init_finalize=[acks[0],acks[-1]],experimental_cadence=2,models_operations=0,frontend_operations=0,capacity_verdict="pending raw audit"))
finally:
 # Existing guards require fresh exact native identities and idle before every policy CAS.
 for node,label,method in [("166","D0_cadence1_restore",controls.update_D0),("167","D1_cadence1_restore",controls.update_D1)]:
  current=controls.policies[node]
  if current!=initial[node]:
   method(dict(initial[node],serial=current["serial"]+1),label)
 controls.verify("restored_policy")
 atomic_json(r/"restore_summary.json",dict(at=utc(),policies=controls.policies,private_policy_writes=sum(controls.policies[k]["serial"]-initial[k]["serial"]for k in initial),model_operations=0,both_same_owned_idle=True,restored_cadence1=True))
print(json.dumps(dict(outputs=s["effective_public_output_tokens"],elapsed_s=s["elapsed_s"],SLO=s["diagnostic_reference_SLO"])))
