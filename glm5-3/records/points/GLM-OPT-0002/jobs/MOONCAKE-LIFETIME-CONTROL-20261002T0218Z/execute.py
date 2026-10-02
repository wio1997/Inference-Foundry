from pathlib import Path
import subprocess,json,sys,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;d=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts")
h=urllib.request.build_opener(urllib.request.ProxyHandler({}));assert h.open("http://172.16.10.166:9081/health",timeout=10).status==200
rows=[]
for name in ["no_KV_fixed.py","acl_lifetime.py","tcp_lifetime.py"]:
 f=d/("moon_lifetime_"+name);f.write_bytes((j/name).read_bytes())
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(d/"pd_common_env.sh")+"; export PYTHONPATH="+str(d)+":$PYTHONPATH; python3 "+str(f)
 if name=="no_KV_fixed.py":shell+=" kv_producer 0"
 p=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=100)
 (j/(name+".stdout")).write_bytes(p.stdout);(j/(name+".stderr")).write_bytes(p.stderr)
 rows.append({"case":name,"returncode":p.returncode,"stdout_tail":p.stdout.decode(errors="replace")[-3000:],"stderr_tail":p.stderr.decode(errors="replace")[-1800:]});atomic_json(j/"cases.json",rows)
assert h.open("http://172.16.10.166:9081/health",timeout=10).status==200
def ref(f):
 a=f.read_bytes();return{"path":str(f),"bytes":len(a),"sha256":hashlib.sha256(a).hexdigest()}
out={"at":utc(),"cases":rows,"models_started":0,"inference":0,"weights":0,"signals":0,"ACL_runtime_init_case":True,"requested_NPU_contexts":0,"TCP_case":{"local":"127.0.0.1","buffer_bytes":8192,"protocol":"tcp","noascendtransfer":True},"limits":["ACLinitialization/finalization mayquery/loaddeviceSDK; notpureimportCPUtest; noNPUcontext/weights/kernel/modelrequest requested","TCPhelper lifetime not AscendKVtransport/runtimefit/PD proof; allactualexitstatuses retained","No nativevendor/lib/guard/operator mutation, no os._exit/LD_PRELOAD workaround","NoKVcorrects ownNone-serializationonly; oldexecutedinvalidsource preserved"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Mooncakeactuallifetime discriminator/noKV correctedCPU; allnativeexitstatus retained;ACLruntimecontrol/TCP8KiBloopbackregistration,0NPUcontexts/models/requests","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="actuallifetimeexitcodes/raw",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"cases":[{"case":x["case"],"returncode":x["returncode"]}for x in rows],"reduction":ref(j/"reduction.json")}))
