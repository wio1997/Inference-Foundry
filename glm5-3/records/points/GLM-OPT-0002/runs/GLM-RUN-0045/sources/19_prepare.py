import pathlib,json,sys,subprocess,shlex,re,urllib.request,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import same_process,atomic_json
from owner_guard import start_watchdog
start_watchdog()
r=pathlib.Path(__file__).parent;old=r.parent/"GLM-RUN-0044"
state=json.loads((old/"state.json").read_text());assert state["status"]=="failed" and not same_process(state["owner"])
audit=json.loads((old/"reduction.json").read_text());assert audit["diagnostic_evidence_valid"]and audit["verdict"]=="REJECT"and audit["inference_attempts"]==2
# Verify exact newly rebuilt45 epoch/STORE1 after configured native rollback; no extra model reload.
exec(compile((r/"adopt_models.py").read_text(),str(r/"adopt_models.py"),"exec"))
ids=json.loads((r/"adopted_model_identities.json").read_text());assert all(all(v[k]==json.loads((r/"startup_model_identities.json").read_text())[n][k]for k in ["host","pid","identity","argv"])for n,v in ids.items())
store={}
for key,item in ids.items():
 node=item["host"];pid=item["pid"];code="import pathlib,json;print(json.dumps([x.decode()for x in pathlib.Path('/proc/"+str(pid)+"/environ').read_bytes().split(bytes([0]))if x.startswith(b'VLLM_ENABLE_RESPONSES_API_STORE=')]))";argv=["python3","-c",code]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(argv)]
 result=subprocess.run(argv,capture_output=True,text=True,timeout=30);result.check_returncode();env=json.loads(result.stdout);assert env==["VLLM_ENABLE_RESPONSES_API_STORE=1"];store[key]=env
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://172.16.10.166:9081/metrics",timeout=10)as reply:text=reply.read().decode()
(r/"prepare_native.metrics").write_text(text)
for key in ["num_requests_running","num_requests_waiting"]:
 vals=re.findall(r"^vllm:"+key+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",text,re.M);assert vals and all(float(x)==0 for x in vals)
ports=subprocess.check_output(["ss","-ltnp"],text=True);assert not re.search(r":8002\s",ports)
atomic_json(r/"native_store.json",store)
print(json.dumps({"native_cohort":"new45-nativeFalse-STORE1","STORE":store,"new_model_requests":0,"native_signals":0}))
