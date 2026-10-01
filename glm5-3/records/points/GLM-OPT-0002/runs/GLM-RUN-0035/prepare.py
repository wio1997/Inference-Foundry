import pathlib,json,sys,hashlib,subprocess,urllib.request,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process,atomic_json,utc
r=pathlib.Path(__file__).parent;old=r.parent/"GLM-RUN-0034"
s=json.loads((old/"state.json").read_text());assert s["status"]=="completed" and not same_process(s["owner"])
assert json.loads((old/"reduction.json").read_text())["measurement_valid"]
cpu=r.parents[1]/"jobs/COUPLED-FAULT-CONTRACTS-20261001T2130Z";data=json.loads((cpu/"reduction.json").read_text());assert data["valid"] and data["cpu_tests"]["tests"]==35
for pin in data["source_identities"]:assert hashlib.sha256(pathlib.Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"]
exec(compile((r/"adopt_models.py").read_text(),str(r/"adopt_models.py"),"exec"))
ident=json.loads((r/"adopted_model_identities.json").read_text());assert ident==json.loads((old/"adopted_model_identities.json").read_text())
epoch=hashlib.sha256(json.dumps({k:{f:v[f]for f in ["host","pid","identity","argv"]}for k,v in ident.items()},sort_keys=True).encode()).hexdigest()
groups=[{"id":"EP32-native-Run34","epoch":epoch,"members":["DP0","DP1"]}]
assert json.loads((r/"execution_groups.json").read_text())==groups
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for rank,node,port in [(0,"166",9081),(1,"167",9900)]:
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=5)as reply:text=reply.read().decode()
 (r/("initial_DP"+str(rank)+".metrics")).write_text(text)
 for key in ["num_requests_running","num_requests_waiting"]:
  vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(v)==0 for v in vals)
p=subprocess.run(["ss","-ltnp"],capture_output=True,text=True);p.check_returncode();assert not re.search(r":8002\s",p.stdout)
atomic_json(r/"prepared_identity.json",{"at":utc(),"exact_native_cohort_retained":True,"native_requests_replayed":0,"groups":groups,"scope":"newgateway actualhealthy capability; nativefailure branch separatelyCPUreplaysactualRun32payload, no additionalOOM"})
