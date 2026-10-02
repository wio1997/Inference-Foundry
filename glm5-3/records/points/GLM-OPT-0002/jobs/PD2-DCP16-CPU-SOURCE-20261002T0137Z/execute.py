from pathlib import Path
import json,sys,subprocess,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;dest=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts/pd2_dcp16_CPU_probe.py");dest.write_bytes((j/"probe.py").read_bytes())
p=subprocess.run(["docker","exec","glm52-single","python3",str(dest)],capture_output=True,timeout=180);(j/"probe.stdout").write_bytes(p.stdout);(j/"probe.stderr").write_bytes(p.stderr);p.check_returncode();cfg=json.loads(p.stdout.decode().splitlines()[-1]);assert len(cfg["rows"])==4 and cfg["weights_loaded"]==cfg["model_requests"]==0
paths=["/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py","/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py","/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py","/vllm-workspace/vllm/vllm/config/kv_transfer.py","/vllm-workspace/vllm/vllm/config/cache.py"]
index=[];(j/"native_sources").mkdir()
for i,path in enumerate(paths):
 raw=subprocess.check_output(["docker","exec","glm52-single","cat",path],timeout=30);f=j/"native_sources"/(str(i)+"_"+Path(path).name);f.write_bytes(raw);lines=raw.decode().splitlines();terms=["decode_context_parallel_size","remote_dcp_size","get_cp_group","kv_cache_memory_bytes","kv_producer"]
 hits=[k for k,x in enumerate(lines)if any(t in x for t in terms)]
 index.append({"path":path,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"snapshot":str(f),"relevant_lines":[{"line":k+1,"context":lines[max(0,k-2):k+4]}for k in hits[:25]]})
atomic_json(j/"source_index.json",index)
out={"at":utc(),"CPU_config":cfg,"native_source_index":{"path":str(j/"source_index.json"),"bytes":(j/"source_index.json").stat().st_size,"sha256":hashlib.sha256((j/"source_index.json").read_bytes()).hexdigest()},"hypothesis":"Prior successfulEP32coupledDP2TP16DCP16 per-rankweights25-26GiB versusolderstockEP16~47GiB changesdualresidentPD feasibility. Available native CP-aware connector metadata may allowseparateproducer/consumer epochs onexisting2hosts. FixedFPoperators; memory/Mooncakebuffers/runtimepeak/metadata/transfer/fallback remainunknown.","limits":cfg["limits"]+["No numericalupperbound or PDstablecapacity claim; speculative weight savings unspecified withoutactualnoMTPprofile"]}
atomic_json(j/"reduction.json",out)
def ref(f):
 raw=f.read_bytes();return{"id":f.name,"path":str(f),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"actualinstalledCPUfull4roleconfig/sourceindex"}
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"4 nativeDP2TP16DCP16EP32 P/D CPUconfigs accepted; producer noMTP/Graph, consumerK5FULL, CP-aware sourcecaptured;0weights/engines/modelrequests, actualfit/transport/fallbackunknown","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ref(j/"reduction.json"),ref(j/"source_index.json")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"configs":cfg["rows"],"weights":0,"inference":0,"reduction":ref(j/"reduction.json")}))
