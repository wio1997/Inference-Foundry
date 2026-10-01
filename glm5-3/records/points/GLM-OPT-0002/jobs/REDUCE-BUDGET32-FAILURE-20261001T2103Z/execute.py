import pathlib,json,hashlib,subprocess,re,sys
from datetime import datetime,timezone
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text());r=pathlib.Path(job["inputs"][0]["path"])
state=json.loads((r/"state.json").read_text());assert state["status"]=="failed" and state["failure_phase"]=="cold" and not same_process(state["owner"])
def ref(p):
 raw=p.read_bytes();return {"path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
spec=json.loads((r/"controller_spec.json").read_text())
for pin in spec["stages"][0]["sources"]:assert hashlib.sha256(pathlib.Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
logs={};metadata={};memory={}
for rank,node in [(0,"166"),(1,"167")]:
 argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/DP_run32_"+str(rank)+".log"]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167"," ".join(argv)]
 p=subprocess.run(argv,capture_output=True,timeout=30);p.check_returncode();dest=r/("DP_run32_"+str(rank)+".log");dest.write_bytes(p.stdout);lines=p.stdout.decode(errors="replace").splitlines()
 logs["DP"+str(rank)]={**ref(dest),"OOM_lines":[{"line":i+1,"text":line}for i,line in enumerate(lines)if "torch.OutOfMemoryError:" in line][:32]}
 metadata["DP"+str(rank)]={"installation":[{"line":i+1,"value":json.loads(line.split("GLM_DP_METADATA_INSTALLED ",1)[1])}for i,line in enumerate(lines)if "GLM_DP_METADATA_INSTALLED " in line],"shapes":[{"line":i+1,"value":json.loads(line.split("GLM_DP_METADATA ",1)[1])}for i,line in enumerate(lines)if "GLM_DP_METADATA {" in line]}
 memory["DP"+str(rank)]=[{"line":i+1,"text":line}for i,line in enumerate(lines)if "Actual usage:" in line or "GPU KV cache size" in line]
assert logs["DP0"]["OOM_lines"] and all(len(v["installation"])==16 for v in metadata.values())
a=json.loads((r/"control_DP0/load_result.json").read_text());assert not a["valid"] and a["effective_output_tokens"]==0 and a["request_count"]==1
row=a["requests"][0];assert row["http_status"]==200 and row["done"] and row["usage"] is None and not row["finish_reasons"] and row["error"]["code"]==500
events=[json.loads(x)for x in (r/"control_DP0/cold_DP0.response.jsonl").read_text().splitlines()];assert events
d1=r/"control_DP1/cold_DP1.response.jsonl";assert d1.exists() and d1.stat().st_size==0 and not (r/"control_DP1/load_result.json").exists()
for rank in [0,1]:assert hashlib.sha256((r/("control_DP"+str(rank))/"request.body.json").read_bytes()).hexdigest()=="349de8820c328303ac846a72906ecf385589b2fb6b65866dfb4b8951251cba9b"
out={"run_id":r.name,"valid_failure_evidence":True,"measurement_valid":False,"functional_acceptance":False,"verdict":"REJECT","scope":"budget16384/gmu.87/K5/nativeDP2TP16DCP16EP32 cold81932->2048","generated_at":utc(),"new_client_attempts":2,"new_completed_inference_requests":0,"new_effective_output_tokens":0,"DP0_outcome":"HTTP200/nativeSSE500/DONE/noUsage/nofinish/zero committed output","DP1_outcome":"issued attempt interrupted by peer controller failure; empty event artifact, outcome unknown/zero credit","native_logs":logs,"metadata":metadata,"native_startup_memory":memory,"source_pins_valid":True,"limits":["Startup/Graph fit does not prove runtime mergedMTP temporary fit","Failed config not hardware bound; lower KV allocation can free runtime workspace","Only two issued attempts, subsequent stage queue not replayed","Metadata ragged idle6 vsmax16384 observed, not speedup certificate"]}
atomic_json(r/"reduction.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,valid_failure_evidence=True,verdict="REJECT",failure_phase="cold",new_client_attempts=2,new_completed_inference_requests=0,new_effective_output_tokens=0,reduction=ref(r/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nREJECT current16384/.87 native coldlong configuration: startup/Graph healthy and32metadata installation receipts, but firstcold mergedMTP DCP path OOM allocating1.01GiB at~1GiBfree. HTTP200 streamSSE500/DONE/noUsage/nofinish, zero credit. Peer attempt interrupted/unknown/zero credit. Two issued attempts, zero completed/outputs; remaining stages unexecuted. Startup weights25.72GiB, activation1.36,KV23.65-23.66; runtime allocated56.12GiB before failing temporary, startup fit is not runtime peak. Metadata idle6/global16384 raggedcontrol observed. Native operator computation unchanged. LowerKV budget is lawful next memory feasibility discriminator; not hardware capacity bound. Rawlogs/body/event/source hashes retained.\n")
p=r/"reduction.json";e=ref(p);e.update(id="reduction",locator="native failure, allissuedattempts and source/memory/metadata evidence")
atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Run32 native16384/.87 runtimeMTP OOM,2issuedattempts0completed0outputs; raw failure evidence retained","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"Startup/Graph fit but runtime mergedMTP temporary OOM; streamnative500 has no usage/finish andzero credit; peer interruptedunknown","scope":{"run_id":r.name,"verdict":"REJECT_specific_config"},"evidence_ids":["reduction"]}],"evidence":[e],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"verdict":out["verdict"],"attempts":2,"outputs":0,"reduction":ref(p)}))
