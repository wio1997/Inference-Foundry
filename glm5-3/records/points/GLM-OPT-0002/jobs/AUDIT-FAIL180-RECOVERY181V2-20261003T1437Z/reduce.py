from pathlib import Path
import sys,json,hashlib,subprocess,re,shlex,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0180";recovery=p/"runs/GLM-RUN-0181";old=p/"runs/GLM-RUN-0179"
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
state=json.loads((r/"state.json").read_text());rest=json.loads((recovery/"state.json").read_text())
assert state["status"]=="failed" and rest["status"]=="completed" and not same_process(state["owner"]) and not same_process(rest["owner"])
spec=json.loads((r/"controller_spec.json").read_text())
for v in spec["stages"][0]["sources"]:
 assert ref(v["path"])["sha256"]==v["sha256"] and Path(v["path"]).read_bytes()==Path(v["snapshot"]).read_bytes()
summary=json.loads((recovery/"recovery_summary.json").read_text());assert summary["valid"] and summary["native32_same_idle"] and summary["native_model_signals"]==0 and summary["model_operations"]==0 and summary["new_inference_requests"]==0 and summary["D0_full_peer_hints_restored"] and summary["bothSTORE_nativewire_same"]
assert "timeout=1700" in (r/"formal.py").read_text() and "1700 seconds" in (r/"formal_experiment.stderr").read_text()
full=json.loads((r/"formal/benchmark/full_execution.json").read_text());assert full["status"]=="cancelled" and full["exit_code"]==-15
assert not (r/"formal_native_formal.stdout").exists()
# Fresh read-only identities; never borrow an old controller guard or signal anything.
roots=json.loads((recovery/"standalone_root_identities.json").read_text());members=json.loads((recovery/"standalone_native_members.json").read_text());physical={}
for key,o in roots.items():
 args=["python3","-c",(recovery/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120)
 (j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 ids={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"] and all(ids[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 physical[key]=dict(root_same=True,NPU16_same=True)
public=json.loads((recovery/"public_service_proof.json").read_text());host=public["host"];assert same_process(host)
assert [v.decode()for v in Path("/proc/"+str(host["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert all(x["active_requests"]==0 and not x["group_faulted"] and x["work_ranking_calibrated"]for x in placement["replicas"])
def metric(f):
 d={}
 for line in Path(f).read_text().splitlines():
  if line and not line.startswith("#"):
   k=line.split("{")[0].split()[0];d[k]=d.get(k,0)+float(line.rsplit(" ",1)[1])
 return d
counters={}
for node,port,serial,threshold in [("166",9081,3,1024),("167",9900,17,4096)]:
 key="D0"if node=="166"else"D1"
 with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10)as res:b=res.read()
 (j/(key+"_final.metrics")).write_bytes(b)
 start=metric(r/("prepare_"+node+".metrics"));end=metric(j/(key+"_final.metrics"))
 assert end["vllm:num_requests_running"]==end["vllm:num_requests_waiting"]==0
 delta={k:end[k]-v for k,v in start.items()if k.endswith("_total")}
 assert delta["vllm:num_preemptions_total"]==0
 plug="/data/tiankuan/wio/glm52-pd/deploy/plugins/"+("local_pp175"if node=="166"else"local_pp168")
 args=["python3","-c","from pathlib import Path;import json;print((Path("+repr(plug)+")/'issue_budget_policy.json').read_text())"]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();pol=json.loads(z.stdout)
 assert pol==dict(schema_version=1,cohort_id="GLM-COHORT-0137",budget_tokens=8192,prefill_threshold_tokens=threshold,prefill_cadence=1,serial=serial)
 counters[key]=dict(delta=delta,idle=True,policy=pol)
cursor=json.loads((r/"formal/trace_cursor.json").read_text())
tracepath=p/"runs/GLM-RUN-0125/router_trace.jsonl"
trace=[json.loads(x)for x in tracepath.read_text().splitlines()if x.strip()]
# Select the six actual requests by exact unique run salt through memo body proof.
memo=[json.loads(x)for x in(old/"token_memo_trace.jsonl").read_text().splitlines()if x.strip()]
rewrites=[]
for x in memo:
 if x.get("event")=="chat_token_cache_native_body":
  body=Path(x["original_body"]["path"]).read_bytes()
  if b"GLM-RUN-0180-dual-PP4-memo-four61440-c4-t1024-"in body:rewrites.append(x)
assert len(rewrites)==6
leases=[];proofs=[]
for x in rewrites:
 body=Path(x["original_body"]["path"]).read_bytes();raw=Path(x["native_body"]["path"]).read_bytes();a=json.loads(body);b=json.loads(raw);ids=b.pop("kv_transfer_params")["prompt_token_ids"];assert b==a and len(ids)==x["native_prompt_tokens"]in [73740,81932]
 assert raw==body.rstrip()[:-1]+b',"kv_transfer_params":{"prompt_token_ids":'+json.dumps(ids,separators=(",",":")).encode()+b'}}'
 source=x["tokenize_source"];tok=json.loads(Path(source["response"]["path"]).read_bytes());assert tok["tokens"]==ids and tok["count"]==len(ids)
 for v in [x["original_body"],x["native_body"],source["response"],source["request"]]:assert ref(v["path"])==v
 matches=[v for v in trace if v["event"]=="lease_acquired" and v.get("body_sha256")==x["original_body_sha256"] and v.get("path")=="/v1/chat/completions" and v["replica"]==("D0"if "166:9081"in x["native_origin"]else"D1") and v["monotonic_ns"]<=x["monotonic_ns"] and x["monotonic_ns"]-v["monotonic_ns"]<10000000000]
 assert len(matches)==1;lease=matches[0];leases.append(lease)
 events=[v for v in trace if v.get("lease_id")==lease["lease_id"]]
 assert len([v for v in events if v["event"]=="lease_released"])==1
 contracts=[v for v in events if v["event"]=="upstream_stream_contract"];assert len(contracts)==1;c=contracts[0]
 assert ref(c["wire_path"])==dict(path=c["wire_path"],bytes=c["wire_bytes"],sha256=c["wire_sha256"])
 data=Path(c["wire_path"]).read_bytes();done=b"data: [DONE]"in data;usage=None;finishes=[];ids_nonnull=0
 for line in data.splitlines():
  if not line.startswith(b"data: ") or line==b"data: [DONE]":continue
  val=json.loads(line[6:])
  if val.get("usage")is not None:usage=val["usage"]
  for choice in val.get("choices",[]):
   if choice.get("finish_reason")is not None:finishes.append(choice["finish_reason"])
   if choice.get("token_ids")is not None:ids_nonnull+=1
 valid=done and usage is not None and usage.get("completion_tokens")==lease["output_budget"] and "length"in finishes
 proofs.append(dict(lease=lease,native_contract=c["contract"],wire=ref(c["wire_path"]),completed_full_contract=valid,done=done,usage=usage,finish_reasons=finishes,output_ID_chunks=ids_nonnull,body_transform_valid=True,memo_result=x["cache_result"],native_prompt_tokens=len(ids),tokenize_source=source))
assert len({x["lease_id"]for x in leases})==6
for budget,each in [(1,1),(61440,2)]:assert {k:sum(v["replica"]==k and v["output_budget"]==budget for v in leases)for k in["D0","D1"]}==dict(D0=each,D1=each)
complete=[x for x in proofs if x["completed_full_contract"]]
credited=sum(x["usage"]["completion_tokens"]for x in complete)
native_total=sum(x["delta"]["vllm:generation_tokens_total"]for x in counters.values())
success=sum(x["delta"]["vllm:request_success_total"]for x in counters.values())
assert success==len(complete) and native_total>=credited
sampling=[]
for key in["D0","D1"]:
 files=sorted((r/"formal").glob("sample*_"+key+".metrics"),key=lambda f:int(f.name.split("_")[0][6:]))
 assert files
 for a,b in zip(files[-7:-1],files[-6:]):
  av=metric(a);bv=metric(b);d={k:bv[k]-v for k,v in av.items()if k.startswith("vllm:spec_decode_")and k.endswith("_total")}
  drafts=d.get("vllm:spec_decode_num_draft_tokens_total",0)
  if drafts>0:sampling.append(dict(domain=key,from_sample=a.name,to_sample=b.name,native_MTP=d,accepted_fraction=d.get("vllm:spec_decode_num_accepted_tokens_total",0)/drafts))
out=dict(at=utc(),run_id=r.name,audit_valid=True,measurement_valid=False,verdict="INVALID",cause="1700s inherited execution-wrapper deadline; full CLI cancelled, restoration rejected while native busy",
 source_pins=len(spec["stages"][0]["sources"]),full_execution=full,physical=physical,public179_same=True,placement=placement,
 actual_requests=proofs,complete_contracts=len(complete),complete_full61440_requests=sum(x["lease"]["output_budget"]==61440for x in complete),
 credited_complete_outputs=credited,total_native_generation_tokens=native_total,partial_or_aborted_generation_cost=native_total-credited,
 native_success_count=success,total_native_counters=counters,late_MTP_samples=sampling,recovery181=ref(recovery/"recovery_summary.json"),
 SDK180_init_finalize="unknown; captured stdout absent",model_operations=0,new_inference_requests=0,
 limits=["Truncated run cannot supply full4 SLO/fullCLI TPS, hardware or capacity upper bound, KEEP or Current",
 "All committed partial/aborted tokens charged; native per-request partial counts unknown; SSE chunk counts are not token counts",
 "Formal streaming output token_ids null; do not claim exact output ID arrays; prompt native token IDs verified separately",
 "K3 low acceptance conditional on forced ignore_eos long-output trajectory, no quantization quality verdict",
 "Recovery181 no signals/new inference/model operations; same32 idle, policy3/17 and full peer hints, both STORE wire retained"])
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Run180 INVALID1700s cutoff: actual complete and partial native work reconciled; 181 recovery healthy same32 and policies3/17; no inference replay",
 execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="deadline-invalid/complete-vs-partial-cost/native32/181recovery")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,verdict="INVALID",complete_full=out["complete_full61440_requests"],credited=credited,native_total=native_total,partial_cost=native_total-credited,evidence=e)))
