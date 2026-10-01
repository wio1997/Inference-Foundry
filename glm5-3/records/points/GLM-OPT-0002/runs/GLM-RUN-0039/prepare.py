import json,sys,hashlib,re,subprocess,urllib.request
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,same_process,utc
r=Path(__file__).parent;old=r.parent/"GLM-RUN-0038"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed" and not same_process(s["owner"])
a=json.loads((old/"reduction.json").read_text());assert a["measurement_valid"] and a["functional_acceptance"] and a["new_completed_inference_requests"]==37
for pin in json.loads((old/"controller_spec.json").read_text())["stages"][0]["sources"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
cpu=r.parents[1]/"jobs/DURABLE-COUPLED-WIRE-CONTRACTS-20261001T2220Z/reduction.json";c=json.loads(cpu.read_text());assert c["valid"] and c["cpu_tests"]["tests"]==42 and c["cpu_tests"]["native_inference_calls"]==0
for pin in c["source_identities"]:assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
assert not (r/"fault_state.json").exists(),"new finitegateway journal required, no overwrite"
atomic_json(r/"startup_model_identities.json",json.loads((old/"startup_model_identities.json").read_text()))
exec(compile((r/"adopt_models.py").read_text(),str(r/"adopt_models.py"),"exec"))
assert json.loads((r/"adopted_model_identities.json").read_text())==json.loads((old/"adopted_model_identities.json").read_text())
assert json.loads((r/"execution_groups.json").read_text())==json.loads((old/"execution_groups.json").read_text())
exec(compile((r/"snapshot.py").read_text(),str(r/"snapshot.py"),"exec"))
for node in ["166","167"]:
 a=json.loads(json.loads((old/(node+"_source.snapshot.json")).read_text())["stdout"]);b=json.loads(json.loads((r/(node+"_source.snapshot.json")).read_text())["stdout"]);assert a==b
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,node,port in [("DP0","166",9081),("DP1","166",9082),("DP2","167",9900),("DP3","167",9901)]:
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=5)as reply:text=reply.read().decode()
 (r/("initial_"+name+".metrics")).write_text(text)
 for key in ["num_requests_running","num_requests_waiting"]:
  vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(v)==0 for v in vals)
assert not re.search(r":8002\s",subprocess.check_output(["ss","-ltnp"],text=True))
atomic_json(r/"prepared_identity.json",{"at":utc(),"healthy_exact_Run38_native_cohort_retained":True,"same_epoch":True,"native_reloads":0,"native_inference_replayed":0,"new_persistent_gateway_only":True})
