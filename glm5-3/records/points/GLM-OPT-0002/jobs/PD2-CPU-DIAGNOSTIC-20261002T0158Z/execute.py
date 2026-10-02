from pathlib import Path
import json,subprocess,hashlib,sys,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;repo=Path("/data/tiankuan/wio/Inference-Foundry");base=j.parent
def ref(p):
 b=p.read_bytes();return {"path":str(p),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
old=[]
for suffix in ["","-v2","-v3"]:
 d=base/("PD2-DCP16-CPU-SOURCE-20261002T0137Z"+suffix)
 files={p.name:ref(p) for p in d.iterdir() if p.is_file()}
 stdout=(d/"probe.stdout").read_text(errors="replace");stderr=(d/"probe.stderr").read_text(errors="replace")
 cfg=None
 for line in reversed(stdout.splitlines()):
  try:
   obj=json.loads(line)
   if isinstance(obj,dict) and "rows" in obj:cfg=obj;break
  except ValueError:pass
 old.append({"job":d.name,"frozen_files":files,"rows_emitted":None if cfg is None else len(cfg["rows"]),"typed_configuration":cfg,"stderr_tail":stderr[-3500:],"classification":"native interleave guard rejected default1 versusblock128" if not suffix else ("GPT dataclass serialization error plus heap teardown anomaly" if suffix=="-v2" else "four typed configuration constructions/assertions/JSON reached, process thenheap teardown failed; no valid execution Result"),"engines":0,"weights":0,"inference":0})
assert old[-1]["rows_emitted"]==4
assert "corrupted size vs. prev_size" in old[-1]["stderr_tail"]
snap=j/"native_sources";snap.mkdir()
root="/vllm-workspace/vllm-ascend/vllm_ascend/"
paths=[root+"distributed/kv_transfer/kv_p2p/mooncake_connector.py",root+"distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py",root+"distributed/kv_transfer/utils/mooncake_transfer_engine.py","/vllm-workspace/vllm/vllm/config/kv_transfer.py","/vllm-workspace/vllm/vllm/config/cache.py",root+"platform.py",root+"ascend_config.py","/vllm-workspace/vllm/vllm/engine/arg_utils.py","/vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py"]
terms=["remote_dcp_size","decode_context_parallel_size","get_cp_group","cp_kv_cache_interleave_size","kv_cache_memory_bytes","kv_producer","max_model_len","gpu_memory_utilization","kv_consumer","do_remote_prefill","is_kv_transfer","cp_size","remote_block_size"]
index=[]
for n,path in enumerate(paths):
 p=subprocess.run(["docker","exec","glm52-single","cat",path],capture_output=True,timeout=30)
 assert p.returncode==0,(path,p.stderr.decode())
 f=snap/(str(n)+"_"+Path(path).name);f.write_bytes(p.stdout)
 lines=p.stdout.decode().splitlines();hits=[i for i,x in enumerate(lines) if any(t in x for t in terms)]
 index.append(dict(native_path=path,**ref(f),hits=[{"line":i+1,"context":lines[max(0,i-2):i+4]} for i in hits[:60]]))
atomic_json(j/"source_index.json",index)
r=repo/"glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0046"
owners=json.loads((r/"adopted_model_identities.json").read_text())
checker='import pathlib,json,sys; o=json.loads(sys.argv[1]);p=pathlib.Path("/proc")/str(o["pid"]);s=(p/"stat").read_text();a=(p/"cmdline").read_bytes().split(bytes([0]));print(json.dumps({"pid":o["pid"],"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"start_ticks":s[s.rfind(")")+2:].split()[19],"argv":[x.decode()for x in a if x]}))'
owned=[]
for node,o in owners.items():
 cmd=["python3","-c",checker,json.dumps(o)]
 if o["host"]=="167":cmd=["ssh","-o","BatchMode=yes","root@172.16.10.167"]+[" ".join("'"+x.replace("'","'\\''")+"'"for x in cmd)]
 p=subprocess.run(cmd,capture_output=True,timeout=30);assert p.returncode==0,p.stderr.decode();a=json.loads(p.stdout)
 assert a["boot_id"]==o["identity"]["boot_id"] and a["start_ticks"]==o["identity"]["start_ticks"] and a["argv"]==o["argv"]
 owned.append(a)
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"
assert not(Path("/proc")/str(state["owner"]["pid"])).exists(),"nativecontroller still exists"
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
health=http.open("http://172.16.10.166:9081/health",timeout=15);assert health.status==200
metrics=http.open("http://172.16.10.166:9081/metrics",timeout=15).read();(j/"current.metrics").write_bytes(metrics)
gauges=[x for x in metrics.decode().splitlines() if x.startswith(("vllm:num_requests_running","vllm:num_requests_waiting"))]
assert gauges and all(float(x.rsplit(" ",1)[-1])==0 for x in gauges),gauges
out={"at":utc(),"prior_jobs":old,"installed_sources":ref(j/"source_index.json"),"current_native_owners":owned,"native_state":state,"health":200,"idle_gauges":gauges,"models_started":0,"model_requests":0,"signals":0,"limits":["prior v3 hasno valid execution Result/process completed assertion; JSON doesnot prove cleanCPUexit","source acceptance no enginefit/transfer/runtime/ABI/performance proof","CPU corruption not yetlocalized; no guardbypass or operator mutation","producer internal81933maxlen requiresfull144384 nativeconsumer localprefill fallback, pending source/E2E","dualresident weights/noMTP memorysaving/runtimecommunication workspaces remainunmeasured"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Readonly PD CPU failure classification/native CP-aware source capture; fourJSONconfigs emitted butheap teardown failure retained; exactRun46owner/nativeidle verified;0signals/starts/inference","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id=p.name,locator="readonlydiagnostic; not nativefit proof",**ref(p)) for p in [j/"reduction.json",j/"source_index.json"]],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"prior_classifications":[[x["job"],x["classification"]]for x in old],"idle":gauges,"sources":len(index),"reduction":ref(j/"reduction.json")}))
