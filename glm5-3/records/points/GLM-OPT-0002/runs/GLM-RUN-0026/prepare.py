import pathlib,json,sys,urllib.request,re,subprocess
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process
r=pathlib.Path(__file__).parent;old=r.parent/"GLM-RUN-0025"
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed" and not same_process(state["owner"])
red=json.loads((old/"reduction.json").read_text());assert red["valid_partial_evidence"] and red["new_client_attempts"]==10 and red["capability_or_dynamic_attempts"]==0
assert json.loads((old/"manifest.json").read_text())["verdict"]=="INVALID"
assert not list((old/"capability").iterdir()) and not (old/"pilot").exists()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for node,port in [("166",9081),("167",9900)]:
 with opener.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=5) as reply:text=reply.read().decode()
 (r/("initial_"+node+".metrics")).write_text(text)
 for key in ["num_requests_running","num_requests_waiting"]:
  vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(v)==0 for v in vals)
p=subprocess.run(["ss","-ltnp"],capture_output=True,text=True);p.check_returncode();assert not re.search(r":8002\s",p.stdout),"diagnostic gateway already owned"
exec(compile((r/"adopt_models.py").read_text(),str(r/"adopt_models.py"),"exec"))
