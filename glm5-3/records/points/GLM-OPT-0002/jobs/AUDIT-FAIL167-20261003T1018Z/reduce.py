from pathlib import Path
import json,sys,subprocess,hashlib,urllib.request,shlex,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0167"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed" and s["failure_phase"]=="fault" and s["completed_stages"]==["prepare"] and not same_process(s["owner"])
m=json.loads((r/"manifest.json").read_text())
for x in m["source"]["source_identities"]:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes() and ref(x["path"])["sha256"]==x["sha256"]
stop=[json.loads(l)for l in(r/"D1_stop.stdout").read_text().splitlines()if l.startswith("{")]
assert stop[-1]["event"]=="cleanup_complete" and stop[-1]["all_original_native_domain_inactive"] and stop[-1]["npu_workers"]==0 and stop[-1]["signals_to_unknown"]==0
roots=json.loads((r/"standalone_root_identities.json").read_text());root=roots["node0"]
z=subprocess.run(["python3","-c",(r/"live_probe.py").read_text()],input=json.dumps(dict(owner=root,NPU_count=16)).encode(),capture_output=True,timeout=90);z.check_returncode();(j/"D0_owner.stdout").write_bytes(z.stdout)
code='import pathlib,subprocess,re,json;raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60);ids=re.findall(r"^\\|\\s+\\d+\\s+\\d+\\s+\\|\\s+(\\d+)\\s+\\|\\s+[^|]+\\|\\s+\\d+\\s+\\|",raw,re.M);assert not ids; sockets=subprocess.check_output(["ss","-ltnp"],text=True);assert not re.search(r":9900\\s",sockets);print(json.dumps(dict(native_NPU_processes=0,native9900_listening=False,npu_smi=raw)))'
z=subprocess.run(["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",code])],capture_output=True,timeout=90);(j/"D1_absent.stdout").write_bytes(z.stdout);(j/"D1_absent.stderr").write_bytes(z.stderr);z.check_returncode()
a=[json.loads(l) for l in(r/"client_survivor.stdout").read_text().splitlines()if l.startswith('{"event":')];assert [x["event"]for x in a]==["task_acl_init","task_acl_finalize"] and all(x["returncode"]==0 for x in a)
v=json.loads((r/"client_survivor_summary.json").read_text());assert not v["valid"] and v["effective_output_tokens"]==0 and len(v["requests"])==1 and v["requests"][0]["status"]==200
trace=p/"runs/GLM-RUN-0125/router_trace.jsonl";events=[json.loads(l)for l in trace.read_text().splitlines()]
header=r.name+"-client_survivor_D0_retained_0";leases=[x for x in events if x["event"]=="lease_acquired" and x.get("request_header_id")==header];assert len(leases)==2
for lease in leases:
 wire=[x for x in events if x["event"]=="upstream_stream_contract" and x.get("lease_id")==lease["lease_id"]];release=[x for x in events if x["event"]=="lease_released" and x.get("lease_id")==lease["lease_id"]]
 assert len(wire)==len(release)==1 and wire[0]["audit_error"] is None and release[0]["released"] and not release[0]["backend_failure"]
 assert json.loads(Path(wire[0]["wire_path"]).read_text())==json.loads((p/"runs/GLM-RUN-0139/base_json.wire").read_text())
atomic_json(j/"retained_GET_trace.json",dict(leases=leases,duplicate_header=True,actual_both_GET200=True,first_client_raw_overwritten_same_label=True,inference_outputs=0))
http=urllib.request.build_opener(urllib.request.ProxyHandler({}));b=http.open("http://172.16.10.166:9081/metrics",timeout=10).read();(j/"D0_terminal.metrics").write_bytes(b)
def metrics(b):
 d={}
 for l in b.decode().splitlines():
  if l and not l.startswith("#"):
   k=l.split("{")[0].split()[0]
   if k in ["vllm:generation_tokens_total","vllm:prompt_tokens_total","vllm:request_success_total","vllm:num_requests_running","vllm:num_requests_waiting"]:d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
 return d
before=metrics((r/"before_166.metrics").read_bytes());after=metrics(b);assert before==after and after["vllm:num_requests_running"]==after["vllm:num_requests_waiting"]==0
h=json.loads(http.open("http://127.0.0.1:8000/healthcheck",timeout=10).read());snap=json.loads(http.open("http://127.0.0.1:8000/control/replicas",timeout=10).read());assert h["request_num"]==0 and next(x for x in snap["replicas"]if x["id"]=="D1")["group_faulted"] and not next(x for x in snap["replicas"]if x["id"]=="D0")["group_faulted"]
out=dict(at=utc(),run_id=r.name,execution_verdict="INVALID",readonly_terminal_acceptance=True,failure="GPT fixture reused same read-only GET header across prepare and fault; len(leases)==1 rejected duplicate2 after successful owned failed166D1 cleanup",source_pins=len(m["source"]["source_identities"]),new_inference_outputs=0,D0_native16_same_owned_idle=True,D1_native166_retired_and_NPU0=True,guard167_new_native_started=False,SDKclient_init_finalize0=True,actual_both_D0_GET200=True,public164_D0_survivor=True,oldD1_sticky_quarantine=True,retained_trace=ref(j/"retained_GET_trace.json"),original_source_raw_preserved=True,limits=["Original prepare client artifact label overwritten by fault probe; native gateway traces retain both exact wire proofs, no inference attempts in thisRun","No new guard correctness or capacity evidence"])
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run167 INVALID duplicate readonly GET test header after successful owned166D1 cleanup. D0 unchanged/native16/SDK0/public survivor; D1 NPU0/9900 absent/no newnative/no inference. Original failure preserved.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="Actual zero-inference failed fixture and owned cleanup")],unknowns=["PP4 guard full functional E2E pending"],decision_request=None,next_check_at=None));print(json.dumps(out))
