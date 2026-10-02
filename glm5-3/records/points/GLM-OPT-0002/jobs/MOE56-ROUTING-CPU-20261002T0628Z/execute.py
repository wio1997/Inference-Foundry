from pathlib import Path
import sys,json,subprocess,shlex,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0056";site=Path("/data/tiankuan/wio/glm52-pd/deploy");plug=site/"plugins/pd_dual_run56";rows=[]
for x in json.loads((r/"planned_launch.json").read_text()):
 if x["node"]!="166":continue
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(plug)+":$PYTHONPATH "+" ".join(k+"="+shlex.quote(v)for k,v in x["env"].items())+"; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(j/"config_probe.py")+" "+shlex.quote(json.dumps(x["argv"]))
 t=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=180);(j/(x["role"]+".stdout")).write_bytes(t.stdout);(j/(x["role"]+".stderr")).write_bytes(t.stderr);t.check_returncode()
 receipts=[json.loads(l.split("ROUTING_RECEIPT ",1)[1])for l in t.stdout.decode().splitlines()if l.startswith("ROUTING_RECEIPT ")];assert len(receipts)==1
 ack=[json.loads(l)for l in t.stdout.decode().splitlines()if l.startswith('{"event": "task_acl_')];assert len(ack)==2 and all(z["returncode"]==0 for z in ack)
 rows+=receipts
assert len(rows)==2
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
native=[]
for n in["ascend_forward_context.py","ops/fused_moe/token_dispatcher.py"]:
 f="/vllm-workspace/vllm-ascend/vllm_ascend/"+n;raw=subprocess.check_output(["docker","exec","glm52-single","cat",f],timeout=20);dest=j/Path(n).name;dest.write_bytes(raw);native.append(dict(native_path=f,**ref(dest)))
limits=["ActualEngineArgs/config/nativecapacity/A3selector onCPU, no NPU group or inference; native per-rank shape/dynamicexecution remains GPU evidence","Nativehierarchy408MB lowerbound observed only maxBs1/h6144; CPUcapacity perTP1 does not certify CANN tiling maxBs for every workload","NominalHCCL env/native route source does not certify physicalallocation/performance/quality/capacity"]
atomic_json(j/"reduction.json",dict(run_id=r.name,rows=rows,native_sources=native,NPU_workers=0,models=0,inference=0,signals=0,limits=limits))
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run56 actualnativeconfig CPU MC2capacity16/perTPrank1; A3selectorMC2<=16/ALLTOALL>16, P/K1 andD/K5FULL6; noNPU/requests/fitproof",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="actualconfig/nativecapacity/A3source selector",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(rows))
