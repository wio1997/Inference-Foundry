from pathlib import Path
import json,hashlib,subprocess,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process
j=Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=Path(job["inputs"][0]["path"]);old=r.parent/"GLM-RUN-0051";s=json.loads((old/"state.json").read_text());assert s["status"]=="cancelled"and not same_process(s["owner"])
a=json.loads((old/"reduction_brief.json").read_text());assert a["measurement_valid"]and a["inference_attempts"]==0 
cfg=json.loads((r.parents[1]/"jobs/PD2-ACL-LIFETIME-CONFIG-20261002T0223Z/reduction.json").read_text());assert cfg["normal_full_config_exit"]and len(cfg["rows"])==4
spec=json.loads((r/"controller_spec.json").read_text());assert [x["id"]for x in spec["stages"]]==["deploy","pilot","finalize"]
for pin in spec["stages"][0]["sources"]:assert Path(pin["path"]).read_bytes()==Path(pin["snapshot"]).read_bytes()and hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
p=subprocess.run(["/usr/bin/python3","/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/controller.py","start",str(r/"controller_spec.json")],capture_output=True,timeout=30);(j/"launch.stdout").write_bytes(p.stdout);(j/"launch.stderr").write_bytes(p.stderr);p.check_returncode()
s=json.loads((r/"state.json").read_text());assert s["status"]=="running"and same_process(s["owner"]);atomic_json(j/"launch_state.json",s)
m=json.loads((r/"manifest.json").read_text());m.update(status="running",controller_state_ref=str(r/"state.json"));atomic_json(r/"manifest.json",m)
def ref(f):
 a=f.read_bytes();return{"id":f.name,"path":str(f),"bytes":len(a),"sha256":hashlib.sha256(a).hexdigest(),"locator":"actualuniquecontrollerlaunchonly"}
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"running","summary":"Run52uniquecontroller launched newnativePDportnamespace, exactknown51P-live/D-failedowner/nativeguards precedecleanup; P/K1eager thenD/K5FULL residentfit/transferpending","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[{"host":s["owner"]["host"],"pid":s["owner"]["pid"],"boot_id":s["owner"]["boot_id"],"start_ticks":s["owner"]["start_ticks"],"readiness":"unready","evidence_ids":["launch_state.json"]}]},"findings":[],"evidence":[ref(j/"launch_state.json")],"unknowns":["Nativefit/HCCL/Graph/64workercomposition/SDKCLI/realCPKVtransfer/fullinputfallback unknown","No oldqueue/publicgateway/KEEP/capacity claim"],"decision_request":None,"next_check_at":None})
print(json.dumps({"controller":s["owner"],"spec":s["spec_sha256"],"pins":len(spec["stages"][0]["sources"]),"actual":"launch_only"}))
