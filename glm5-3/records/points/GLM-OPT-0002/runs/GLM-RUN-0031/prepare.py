import pathlib,json,sys,subprocess,shlex,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process,atomic_json
r=pathlib.Path(__file__).parent;old=r.parent/"GLM-RUN-0030"
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed" and not same_process(state["owner"])
assert json.loads((old/"reduction.json").read_text())["valid_partial_evidence"]
exec(compile((r/"adopt_models.py").read_text(),str(r/"adopt_models.py"),"exec"))
ids=json.loads((r/"adopted_model_identities.json").read_text());store={}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for key,item in ids.items():
 node=item["host"];pid=item["pid"]
 code="import pathlib,json;p=pathlib.Path('/proc/"+str(pid)+"/environ');print(json.dumps([x.decode()for x in p.read_bytes().split(bytes([0]))if x.startswith(b'VLLM_ENABLE_RESPONSES_API_STORE=')]))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);p.check_returncode();env=json.loads(p.stdout);assert not env or env==["VLLM_ENABLE_RESPONSES_API_STORE=0"],"store enabled; revise unexecuted request assumptions";store[key]={"native_env":env,"source_default":False}
 with opener.open("http://172.16.10."+node+":"+("9081"if node=="166"else"9900")+"/metrics",timeout=5) as reply:text=reply.read().decode()
 (r/("initial_"+key+".metrics")).write_text(text)
 for name in ["num_requests_running","num_requests_waiting"]:
  values=re.findall(r"^vllm:"+name+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert values and all(float(x)==0 for x in values)
p=subprocess.run(["ss","-ltnp"],capture_output=True,text=True);p.check_returncode();assert not re.search(r":8002\s",p.stdout)
atomic_json(r/"native_store.json",store)
