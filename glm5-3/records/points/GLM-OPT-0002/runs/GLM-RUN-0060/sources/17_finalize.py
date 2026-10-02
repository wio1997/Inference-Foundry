from pathlib import Path
import json,sys,hashlib,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog
start_watchdog()
r=Path(__file__).parent;summary=json.loads((r/"pilot_summary.json").read_text());assert summary["measurement_valid"]
for x in json.loads((r/"planned_launch.json").read_text()):
 args=["cat",x["log"]]
 if x["node"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 p=subprocess.run(args,capture_output=True,timeout=60);p.check_returncode();(r/("final_"+x["role"]+"_"+x["node"]+".native.log")).write_bytes(p.stdout)
art=[]
for f in r.rglob("*"):
 if not f.is_file()or"__pycache__"in str(f)or f.name in ["artifact_index.json","manifest.json","state.json","controller.log"]:continue
 z=f.read_bytes();art.append({"path":str(f),"bytes":len(z),"sha256":hashlib.sha256(z).hexdigest()})
atomic_json(r/"artifact_index.json",art)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",valid=True,verdict="INCONCLUSIVE",at=utc(),measurement=summary,artifacts=art,limits=summary["limits"]+["IndependentactualnativeKVcompletion/roles/metadata/operators/effectivecounteraudit pending"]);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nDirectnative dualresidentPD functionalpilotcompleted. Elevennativeinferences/completed356nativecommits; sevenpublicrequests352effectiveoutputs/fourPhelpers4separateinternalcommits. NativeCPmetadata/P→Dtwooppositerankdirections/Dlocalfullinput>81933 actualnativeguards preserved; independentauditpending. No publicgateway/fullcompatibility/KEEP/stablecapacity claim. Allraw/source/owners/counters retained.\n")
print(json.dumps({"native_attempts":11,"public_effective_output":352,"independent_audit":"pending"}))
