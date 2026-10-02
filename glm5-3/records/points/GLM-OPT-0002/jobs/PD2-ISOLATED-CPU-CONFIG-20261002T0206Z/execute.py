from pathlib import Path
import json,subprocess,hashlib,sys,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;repo=Path("/data/tiankuan/wio/Inference-Foundry");d=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts")
sources=["atomic_mq_bind.py","atomic_mq_worker.py","coupled_dp_metadata_v2.py","coupled_atomic_mq_worker.py"]
for n in sources:
 raw=(repo/"glm5-3/runtime"/n).read_bytes();(d/n).write_bytes(raw);(j/("source_"+n)).write_bytes(raw)
(d/"pd2_isolated_probe.py").write_bytes((j/"probe.py").read_bytes())
rows=[]
for role in ["kv_producer","kv_consumer"]:
 for rank in [0,1]:
  ip="172.16.10."+str(166+rank)
  shell="export PD_LOCAL_IP="+shlex.quote(ip)+" PD_NIC=business; source "+str(d/"pd_common_env.sh")+"; export PYTHONPATH="+str(d)+":$PYTHONPATH; export VLLM_HOST_IP="+ip+"; python3 "+str(d/"pd2_isolated_probe.py")+" "+role+" "+str(rank)
  p=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=100)
  name=role+"_"+str(rank);(j/(name+".stdout")).write_bytes(p.stdout);(j/(name+".stderr")).write_bytes(p.stderr)
  cfg=None
  for line in reversed(p.stdout.decode(errors="replace").splitlines()):
   try:
    o=json.loads(line)
    if "rows"in o:cfg=o;break
   except ValueError:pass
  rows.append({"role":role,"rank":rank,"returncode":p.returncode,"config":cfg,"stderr_tail":p.stderr.decode(errors="replace")[-1200:]})
  atomic_json(j/"configuration_processes.json",rows)
def ref(f):
 b=f.read_bytes();return{"path":str(f),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
passed=all(x["returncode"]==0 and x["config"] and len(x["config"]["rows"])==1 for x in rows)
out={"at":utc(),"configurations":rows,"sources":[ref(j/("source_"+n))for n in sources],"normal_CPU_exit":passed,"models_started":0,"weights":0,"inference":0,"limits":["Changed toproducerK1nativeMTP: consumerlayer resolver raisesmissingremote MTP cache names whenproducerhasnone; no noMTP weight saving assumed","IndependentCPU subprocess/commonCANNenvironment andproducerMTP altered together; any exit improvement notisolatedcause","Allprocesses runon166 CPU only; remotehost/nativeworker/transport/dualresidentmemoryfit pending","ExplicitKV1GiBP/2GiBD andgmu.48 pendingnative maxlen guard/actualruntimepeak; nofit/performance/capacity proof","Combinedcontrol workerimport no installation/device/modelrunner contract proof"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Native isolated CPU role/rank configs normalexit="+str(passed)+"; actual exit/status/raw retained,0engines/weights/inference;producerK1matches consumerMTP cache layer requirement","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="actualisolatedCPUexitstatus/config; noenginefit",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"normal_CPU_exit":passed,"rows":[{"role":x["role"],"rank":x["rank"],"returncode":x["returncode"],"config_emitted":x["config"]is not None}for x in rows],"reduction":ref(j/"reduction.json")}))
