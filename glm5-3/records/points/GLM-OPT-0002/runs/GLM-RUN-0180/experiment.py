from pathlib import Path
import json,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
import controls
start_watchdog();r=Path(__file__).parent
LOG={"166":"/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp175_166.log","167":"/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp168_167.log"}
def remote(node,code,stem):
 guard();args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);(r/(stem+".stdout")).write_bytes(z.stdout);(r/(stem+".stderr")).write_bytes(z.stderr);z.check_returncode();return z.stdout
offsets={}
for node,path in LOG.items():
 offsets[node]=json.loads(remote(node,"from pathlib import Path;import json;print(json.dumps(dict(path="+repr(path)+",bytes=Path("+repr(path)+").stat().st_size)))","policy_log_offset_"+node))
 atomic_json(r/("policy_log_offset_"+node+".json"),offsets[node])
try:
 controls.update_D1(dict(controls.policies["167"],prefill_threshold_tokens=1024,serial=16),"D1_threshold1024")
 for window in ["formal"]:
  before=json.loads((r.parent/"GLM-RUN-0179/token_memo_stats.json").read_text());atomic_json(r/(window+"_memo_before.json"),before)
  z=subprocess.run([sys.executable,str(r/"formal.py"),window],capture_output=True,timeout=1750)
  (r/(window+"_experiment.stdout")).write_bytes(z.stdout);(r/(window+"_experiment.stderr")).write_bytes(z.stderr);z.check_returncode()
  after=json.loads((r.parent/"GLM-RUN-0179/token_memo_stats.json").read_text());assert after["audit_errors"]==after["lookup_errors"]==0 and after["pending"]==0 and after["cache_bytes"]<=after["cache_byte_budget"]
  atomic_json(r/(window+"_memo_after.json"),after)
 selected={}
 for node,path in LOG.items():
  raw=remote(node,"from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(path)+").read_bytes()["+str(offsets[node]["bytes"])+":])","native_policy_window_"+node)
  (r/("native_policy_window_"+node+".log")).write_bytes(raw)
  states=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in raw.decode().splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
  if node=="166"and not states:
   allraw=remote(node,"from pathlib import Path;import sys;sys.stdout.buffer.write(Path("+repr(path)+").read_bytes())","D0_current_consumption")
   prior=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in allraw.decode().splitlines()if "GLM_ISSUE_BUDGET_SELECTED "in l]
   assert prior;states=prior[-1:]
   atomic_json(r/"D0_consumption_reuse.json",dict(source="latestactualD0SELECTED beforeformal window; unchangedpolicy no new marker expected",selected=states,window_new_selected=False))
  serial=3 if node=="166"else 16
  assert states and all(x["serial"]==serial and x["budget_tokens"]==8192 and x["prefill_threshold_tokens"]==1024 and not x["fallback"]and x["native_max"]==8192 for x in states)
  selected[node]=states
 atomic_json(r/"policy_consumption.json",dict(at=utc(),native_selected=selected,allocated8192_unchanged=True,operator_math_changes=0,D0_policy_unchanged=True))
finally:
 if controls.policies["167"]["serial"]==16:controls.update_D1(dict(controls.policies["167"],prefill_threshold_tokens=4096,serial=17),"D1_threshold4096_restore")
 z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"epoch_terminal"],capture_output=True,timeout=240)
 (r/"epoch_terminal.stdout").write_bytes(z.stdout);(r/"epoch_terminal.stderr").write_bytes(z.stderr);z.check_returncode()
 atomic_json(r/"restore_summary.json",dict(at=utc(),D0_policy_unchanged=controls.policies["166"],D1_policy_restored=controls.policies["167"],private_policy_writes=sum((r/(s+"_ack.json")).exists()for s in["D1_threshold1024","D1_threshold4096_restore"]),model_operations=0,native32_same_idle=True,logical_D0_peer_restored_by_client=True))
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",results={w:json.loads((r/(w+"_wrapper_summary.json")).read_text())for w in["formal"]});atomic_json(r/"manifest.json",m)
