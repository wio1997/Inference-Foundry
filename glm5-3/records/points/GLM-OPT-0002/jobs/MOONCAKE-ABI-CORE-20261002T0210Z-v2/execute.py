from pathlib import Path
import subprocess,json,hashlib,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;old=j.parent/"MOONCAKE-ABI-CORE-20261002T0210Z";o=json.loads((old/"reduction.json").read_text());assert o["origin_core_PID"]==3012814
pid=subprocess.check_output(["docker","inspect","-f","{{.State.Pid}}","glm52-single"]).decode().strip();assert pid.isdigit()
p=subprocess.run(["eu-stack","--core",str(old/"cpu_probe.core"),"--executable","/proc/"+pid+"/root/usr/local/python3.12.13/bin/python3.12","-n","32","-m","-b"],capture_output=True,timeout=100)
(j/"native_stack.txt").write_bytes(p.stdout+p.stderr)
def ref(f):
 a=f.read_bytes();return{"path":str(f),"bytes":len(a),"sha256":hashlib.sha256(a).hexdigest()}
out={"at":utc(),"original_reduction":ref(old/"reduction.json"),"raw_core":o["raw_core"],"old_unwind":"GPT -1CLIinvalid: eu-stack -1 requireslive-p andcannotselectsinglethreadcore; no oldstackproof","stack_returncode":p.returncode,"stack":ref(j/"native_stack.txt"),"same_mooncake_libraries":o["same_mooncake_sharedlibs"],"engines":0,"weights":0,"inference":0,"signals":0,"limits":["ExistingCPUcore only; actualunwind maypartial/missingcontainerELFs, noABI/runtimefit proof","Nativebinary/guards/operators unchanged"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"ExistingCPUcore unwind correctedCLI/no newmodels; actualstackexit/output retained, noABI success assumption","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="existingcore correctedunwindCLI",**ref(j/"reduction.json")),dict(id="stack",locator="nativeunwind actualoutput",**ref(j/"native_stack.txt"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"exit":p.returncode,"stack":ref(j/"native_stack.txt")}))
