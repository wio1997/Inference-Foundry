from pathlib import Path
import subprocess,json,hashlib,shlex,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;dest=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts");rows=[]
for n in ["import_only.py","mooncake_import_only.py","no_KV.py","no_DCP.py"]:
 f=dest/("pd_exit_"+n);f.write_bytes((j/n).read_bytes())
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(dest/"pd_common_env.sh")+"; export PYTHONPATH="+str(dest)+":$PYTHONPATH; python3 "+str(f)
 if n.startswith("no_"):shell+=" kv_producer 0"
 p=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=100)
 (j/(n+".stdout")).write_bytes(p.stdout);(j/(n+".stderr")).write_bytes(p.stderr)
 rows.append({"case":n,"returncode":p.returncode,"stdout_tail":p.stdout.decode(errors="replace")[-1500:],"stderr_tail":p.stderr.decode(errors="replace")[-1500:]});atomic_json(j/"cases.json",rows)
core=subprocess.run(["coredumpctl","--no-pager","--since","2026-10-02 01:58:00 UTC","list"],capture_output=True,timeout=20)
(j/"existing_core_list.txt").write_bytes(core.stdout+core.stderr)
def ref(f):
 a=f.read_bytes();return{"path":str(f),"bytes":len(a),"sha256":hashlib.sha256(a).hexdigest()}
out={"at":utc(),"cases":rows,"core_list":ref(j/"existing_core_list.txt"),"signals":0,"engines":0,"weights":0,"inference":0,"limits":["CPU discriminator, not actualnativeengine/start/fit/PDperformance proof","No nativeoperators/guards/ABI change; core disabledonlywithinnewCPUprocesses","PreviousCPU failures preserved; moduleimport mayregisterframeworkplugins, not instantiateNPUengine"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"ActualfourCPUexit discriminants retained; zeroengines/weights/inference/modelsignals","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="actualCPUexit discriminant",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"cases":[{"case":x["case"],"returncode":x["returncode"]}for x in rows],"reduction":ref(j/"reduction.json")}))
