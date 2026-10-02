from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from owner_guard import start_watchdog,guard
from phase_runner import atomic_json,utc
start_watchdog();r=Path(__file__).parent
shell="export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/dp_run79/native_acl_lifecycle.py script "+str(r/"responses.py")
import controls
controls.policy=json.loads((r/"effective_policy.json").read_text())
try:
 z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=2100)
 (r/"native_full_api.stdout").write_bytes(z.stdout);(r/"native_full_api.stderr").write_bytes(z.stderr);z.check_returncode()
 acks=[json.loads(x)for x in z.stdout.decode().splitlines()if x.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
 s=json.loads((r/"responses_summary.json").read_text());assert s["valid"]and s["new_effective_output_tokens"]==256
 ts=[json.loads((r/("tools_"+label)/"summary.json").read_text())for label in["gateway_first","gateway_second"]];assert all(x["valid"]and len(x["requests"])==4 for x in ts)
 out=dict(at=utc(),functional_acceptance=True,effective_public_output_tokens=256+sum(t["effective_output_tokens"]for t in ts),Responses=s,tool_outputs=sum(t["effective_output_tokens"]for t in ts),tool_cases=8,models_started=0,model_signals=0,limits=["Finite functional gateway/2nativeSTOREdomains E2E, not formalSLO/stablecapacity/KEEP/PD","Independentraw/source/epochs/toolowner/counters audit pending","CPUcustomIDconcurrency fixed; actualnative duplicate-ID concurrent acceptance untested"])
 atomic_json(r/"full_api_summary.json",out);m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results=out);atomic_json(r/"manifest.json",m);print(json.dumps(out))

finally:
 controls.policy=json.loads((r/"effective_policy.json").read_text())
 controls.update(dict(controls.policy,prefill_threshold_tokens=2048,serial=5),"restore_threshold2048")
