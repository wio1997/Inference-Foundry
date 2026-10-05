from pathlib import Path
import sys,json,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0218"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])and s["completed_stages"]==["prepare"]
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"];assert len(pins)==174
for st in spec["stages"]:assert st["sources"]==pins
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(x["path"])["sha256"]==x["sha256"]
root=json.loads((r/"restored/startup_root.json").read_text());assert not same_process(dict(pid=root["pid"],**root["identity"]))
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60);(j/"166.npu_smi.stdout").write_text(raw)
assert not re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+[^|]+\|\s+\d+\s+\|",raw,re.M)
native=Path("/data/tiankuan/wio/glm52-pd/deploy/logs/local_pp218_166.log");(j/"native218.stdout").write_bytes(native.read_bytes());text=native.read_text()
assert "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1"in text and "ValidationError"in text and "Loading model weights"not in text
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",(r/"live_probe.py").read_text()])]
z=subprocess.run(args,input=json.dumps(dict(owner=roots["node1"],NPU_count=16)).encode(),capture_output=True,timeout=120);z.check_returncode();(j/"D1.owner.stdout").write_bytes(z.stdout);v=json.loads(z.stdout)
assert v["npu_worker_pids"]==members["node1"]["npu_worker_pids"]
exp={x["pid"]:x["identity"]for x in members["node1"]["owned_targets"]};assert all(exp[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
def metric(b):
 out={}
 for l in b.decode().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{")[0].split()[0]
  if k.startswith("vllm:"):out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://172.16.10.167:9900/metrics",timeout=15)as res:b=res.read()
(j/"D1.metrics").write_bytes(b);before=metric((r/"before_167.metrics").read_bytes());after=metric(b)
assert all(after[k]==v for k,v in before.items()if k.endswith("_total"))and after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0
with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as res:placement=json.loads(res.read())
by={x["id"]:x for x in placement["replicas"]};assert by["D0"]["group_faulted"]and not by["D1"]["group_faulted"]and all(not x["active_requests"]for x in placement["replicas"])
public=json.loads((p/"runs/GLM-RUN-0217/restored/public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]
# Snapshot raw217 CANN plog and relevant immutable native source, never print wholesale.
code="from pathlib import Path;import json;files=['/root/ascend/log/debug/plog/plog-1535589_20261005094609512.log','/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/sfa_cp.py','/vllm-workspace/vllm-ascend/vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py','/vllm-workspace/vllm-ascend/vllm_ascend/worker/v2/attn_utils.py'];print(json.dumps({f:Path(f).read_text()for f in files}))"
z=subprocess.run(["docker","exec","glm52-single","python3","-c",code],capture_output=True,timeout=60);z.check_returncode();files=json.loads(z.stdout);sources=[]
for i,(name,content)in enumerate(files.items()):
 f=j/("%02d_"%i+Path(name).name);f.write_text(content);sources.append(dict(origin=name,**ref(f)))
out=dict(at=utc(),run_id=r.name,verdict="INVALID",source_count=174,config_rejected_before_weights=True,blocking_Graph_native_guard_preserved=True,new_native_inference=0,new_native_workers=0,D0_NPU_processes=0,D1_native16_retained_allcounters=True,public217_retained=True,root_inactive=root,native_error=ref(j/"native218.stdout"),native_sources=sources,Current=None,limits=["218 rejectedblocking+Graph nativeguard; no V2runtime diagnosis/function/performance evidence","217CANNplog confirmsIndexCheck originaloperation source unknown; SFA_DCP flattenedblocktable footprint3*1128 matches3384 butvalueinterpretation hypothesis","Nextlegal eagerblocking or sameGraph scopedmetadata diagnostic; no guard/operator bypass"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="218 configguard blocking+ACLGraph rejected beforeweights/inference, frozen174/noNPU/retainedD1-public217 readonlyaudit VALID; plog/nativeSFA source snapshot",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="frozen174-nativeconfigguard-zeroNPU-D1retained-plog-source")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(audit_VALID=True,configINVALID=True,D0_NPU0=True,D1native16=True)))
