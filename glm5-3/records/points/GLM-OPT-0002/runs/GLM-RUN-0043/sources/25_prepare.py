import json,sys,urllib.request,re
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
from owner_guard import guard
root=Path(__file__).parent;old=root.parent/"GLM-RUN-0042";guard()
s=json.loads((old/"state.json").read_text());a=json.loads((old/"reduction.json").read_text())
assert s["status"]=="completed" and not same_process(s["owner"]) and a["measurement_valid"]and a["functional_acceptance"] and a["new_client_attempts"]==22
exec(compile((root/"adopt_models.py").read_text(),str(root/"adopt_models.py"),"exec"))
assert json.loads((root/"adopted_model_identities.json").read_text())==json.loads((old/"adopted_model_identities.json").read_text())
assert json.loads((root/"execution_groups.json").read_text())==json.loads((old/"execution_groups.json").read_text())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://172.16.10.166:9081/metrics",timeout=5)as response:raw=response.read()
(root/"before_native.metrics").write_bytes(raw)
for k in ["num_requests_running","num_requests_waiting"]:
 vals=re.findall(r"^vllm:"+k+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M);assert vals and all(float(t)==0 for t in vals)
atomic_json(root/"reuse.json",{"at":utc(),"source_run":old.name,"source_reduction_sha256":__import__("hashlib").sha256((old/"reduction.json").read_bytes()).hexdigest(),"reused_native_cohort":True,"new_model_starts":0,"new_model_signals":0,"reused_tools_capability_dynamic_prefix_new_credit":0,"new_work":"one canonical AISBench round:4 full81932->61440 requests, closedconcurrency2 +oneprefixwarmup73740->1; single TP32DCP1EP32 native208768KV planning tokens. NearKV prediction assumes common prefix block sharing; verify actualpreempt/memory/counters. Finite package not stablecapacity or repeatedKEEP."})
