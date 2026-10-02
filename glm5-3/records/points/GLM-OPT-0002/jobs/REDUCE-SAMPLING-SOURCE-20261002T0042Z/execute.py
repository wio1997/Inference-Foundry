import json,subprocess,hashlib,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;site=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts");dest=site/"reduce_sampling_fullconfig_probe.py";dest.write_bytes((j/"probe.py").read_bytes())
p=subprocess.run(["docker","exec","-e","PYTHONPATH="+str(site),"glm52-single","python3",str(dest)],capture_output=True,timeout=120)
(j/"probe.stdout").write_bytes(p.stdout);(j/"probe.stderr").write_bytes(p.stderr);p.check_returncode();cfg=json.loads(p.stdout.decode().splitlines()[-1])
assert cfg["model_requests"]==cfg["weights_loaded"]==0 and all(x["enable_reduce_sample"]and x["lmhead_tp"]==0 for x in cfg["full_native_engine_config"])
paths=["/vllm-workspace/vllm-ascend/vllm_ascend/"+x for x in ["ascend_config.py","ops/vocab_parallel_embedding.py","sample/sampler.py","sample/rejection_sampler.py","spec_decode/llm_base_proposer.py","worker/model_runner_v1.py","model_executor/warmup/rejection_sampler_triton_warmup.py"]]+["/vllm-workspace/vllm/vllm/config/speculative.py"]
sources=[];snap=j/"native_sources";snap.mkdir()
for i,path in enumerate(paths):
 raw=subprocess.check_output(["docker","exec","glm52-single","cat",path],timeout=30);target=snap/f"{i:02d}_{Path(path).name}";target.write_bytes(raw)
 lines=raw.decode().splitlines();hits=[k for k,x in enumerate(lines)if "enable_reduce_sample"in x]
 sources.append({"path":path,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"snapshot":str(target),"relevant_source_lines":[{"line":k+1,"context":lines[max(0,k-3):k+7]}for k in hits[:12]]})
atomic_json(j/"source_index.json",sources)
out={"at":utc(),"cpu_only":True,"model_calls":0,"weights_loaded":0,"engine_config":cfg,"source_index":{"path":str(j/"source_index.json"),"bytes":(j/"source_index.json").stat().st_size,"sha256":hashlib.sha256((j/"source_index.json").read_bytes()).hexdigest()},"facts":["Installed typed Ascend enable_reduce_sample option defaults false; current42/43 native rejection sampler log records fallback false/no targetindices","Native option supports configuredDP1TP32DCP1EP32 fullCPU configs with finegrainedlmheadTP0 and resolved nativeMTP method recorded; no engine constructed","Native vocab logits gather suppressed when optionenabled, distributed sampler chooses nativeglobal candidates/greedy argmax; existing native implementation only","Installed nativeguards explicitly reject finegrainedlmheadTP override and PD kv_producer; no bypass"],"hypothesis":"Available native distributed sampling control may reduce decode-chain full-vocab collective costs on currentTP32. Need actual native functional allsampling/logprobs/tool/Responses contracts and E2E; source doesnotestablish hotspot/speedup.","limits":["CPU/source only, no GPU tensors/weights or execution of native reduce-sampling path","No model/quantization accuracy comparison or operator implementation edit","No performance/KEEP/hardware limit; native experimental option acceptance is not runtime correctness"]}
atomic_json(j/"reduction.json",out)
def ref(path):
 raw=path.read_bytes();return {"id":path.name,"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"actual CPU fullEngineArgs and installed native distributed sampling source index"}
job=json.loads((j/"job.json").read_text());atomic_json(job["result"]["path"],{"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Installed native reduce-sampling fullTP32DCP1 CPU configs/source captured;0weights/modelcalls, nativepath/E2E pending","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[ref(j/"reduction.json"),ref(j/"source_index.json")],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"config_accepted":True,"rows":[{"rank":x["rank"],"method":x["resolved_spec_method"],"reduce":x["enable_reduce_sample"],"dtype":x["dtype"]}for x in cfg["full_native_engine_config"]],"model_calls":0,"reduction":ref(j/"reduction.json")}))
