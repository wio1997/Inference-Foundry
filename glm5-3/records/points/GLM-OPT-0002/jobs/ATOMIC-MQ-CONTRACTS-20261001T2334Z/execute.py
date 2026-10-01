import json,subprocess,hashlib,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
j=Path(__file__).parent;g=j.parents[4];site=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts")
for s in [g/"runtime/atomic_mq_bind.py",j/"probe.py"]:
 (site/s.name).write_bytes(s.read_bytes())
p=subprocess.run(["docker","exec","-e","PYTHONPATH="+str(site),"glm52-single","python3",str(site/"probe.py")],capture_output=True,timeout=120)
(j/"probe.stdout").write_bytes(p.stdout);(j/"probe.stderr").write_bytes(p.stderr);p.check_returncode()
out=json.loads(p.stdout.decode().splitlines()[-1]);assert out["native_queue_payload_roundtrips"]==32 and out["forced_native_EADDRINUSE_reproduced"]
atomic_json(j/"reduction.json",out)
def ref(path):
 raw=path.read_bytes();return {"id":path.name,"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"actual CPU native MessageQueue occupied-port negative and 32 concurrent atomic endpoint native payload roundtrips"}
job=json.loads((j/"job.json").read_text())
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Native queue CPU contract: original occupied-port rejection, 32 concurrent atomic bind native payload roundtrips, source unchanged","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ref(j/"reduction.json"),ref(j/"probe.stdout")],"unknowns":["CPU loopback contract does not establish multi-host native startup or model performance"],"decision_request":None,"next_check_at":None})
print(json.dumps(out))
