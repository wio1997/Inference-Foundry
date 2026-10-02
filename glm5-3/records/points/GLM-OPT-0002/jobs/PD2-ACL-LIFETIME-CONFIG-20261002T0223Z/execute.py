from pathlib import Path
import subprocess,json,hashlib,sys,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;repo=Path("/data/tiankuan/wio/Inference-Foundry");d=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts")
rows=[]
for node in ["166","167"]:
 def cmd(args,timeout=100):
  if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  return subprocess.run(args,capture_output=True,timeout=timeout)
 for n in ["atomic_mq_bind.py","atomic_mq_worker.py","coupled_dp_metadata_v2.py","coupled_atomic_mq_worker.py","native_acl_lifecycle.py"]:
  source=repo/"glm5-3/runtime"/n
  if node=="166":(d/n).write_bytes(source.read_bytes())
  else:subprocess.run(["scp","-q",str(source),"root@172.16.10.167:"+str(d/n)],check=True,capture_output=True)
  p=cmd(["sha256sum",str(d/n)]);assert p.returncode==0 and p.stdout.decode().split()[0]==hashlib.sha256(source.read_bytes()).hexdigest()
 dest=d/"pd2_acl_config_probe.py"
 if node=="166":dest.write_bytes((j/"probe.py").read_bytes())
 else:subprocess.run(["scp","-q",str(j/"probe.py"),"root@172.16.10.167:"+str(dest)],check=True,capture_output=True)
 rank=0 if node=="166"else 1
 for role in ["kv_producer","kv_consumer"]:
  shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10."+node+" PD_NIC=business; source "+str(d/"pd_common_env.sh")+"; export PYTHONPATH="+str(d)+":$PYTHONPATH; export VLLM_HOST_IP=172.16.10."+node+"; python3 "+str(d/"native_acl_lifecycle.py")+" script "+str(dest)+" "+role+" "+str(rank)
  p=cmd(["docker","exec","glm52-single","bash","-c",shell]);name=node+"_"+role;(j/(name+".stdout")).write_bytes(p.stdout);(j/(name+".stderr")).write_bytes(p.stderr)
  cfg=None;events=[]
  for line in p.stdout.decode(errors="replace").splitlines():
   try:
    o=json.loads(line)
    if "rows"in o:cfg=o
    if o.get("event")in ["task_acl_init","task_acl_finalize"]:events.append(o)
   except ValueError:pass
  rows.append({"node":node,"role":role,"rank":rank,"returncode":p.returncode,"config":cfg,"ACL_events":events,"stderr_tail":p.stderr.decode(errors="replace")[-1500:]});atomic_json(j/"configurations.json",rows)
passed=all(x["returncode"]==0 and x["config"]and len(x["config"]["rows"])==1 and len(x["ACL_events"])==2 and all(z["returncode"]==0 for z in x["ACL_events"])for x in rows)
def ref(f):
 a=f.read_bytes();return{"path":str(f),"bytes":len(a),"sha256":hashlib.sha256(a).hexdigest()}
out={"at":utc(),"normal_full_config_exit":passed,"rows":rows,"wrapper":ref(j/"native_acl_lifecycle.py"),"models_started":0,"weights":0,"inference":0,"signals":0,"NPU_contexts_requested":0,"limits":["ACLSDK initialization/finalization nativeprocesscontrol; no devicecontext/weights/inference requested","Same fourP/D configs/nativeguards/operators; actualnativeentry/dualresidentmemory/connectortransfer stillpending","Taskwrapper callsunmodifiednativeCLI andfinalizesSDK onreturn; nativeengine cleanup/lifecycle stillneedsE2E","Sourceimport/call semantics tested as script mode; actual CLI-mode launch pending"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Bothhost nativeP/D fullconfigs ACLlifetime normal_exit="+str(passed)+"; zero models/contexts/inference; actualnativefit/transportpending","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="actualbothhostnativeconfig/SDKlifetime",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"normal_full_config_exit":passed,"rows":[{"node":x["node"],"role":x["role"],"returncode":x["returncode"]}for x in rows],"reduction":ref(j/"reduction.json")}))
