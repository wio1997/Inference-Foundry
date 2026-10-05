from pathlib import Path
import sys,json,subprocess,shlex,urllib.request,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());out={}
label=sys.argv[1]if len(sys.argv)>1else"epoch"
for key,o in roots.items():
 guard();args=["python3","-c",(r/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100);(r/(label+"_"+key+".stdout")).write_bytes(z.stdout);(r/(label+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 out[key]=dict(root_same=True,NPU16_same=True)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for node,port in[("166",9081)]:
 with http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=10)as response:b=response.read()
 (r/(label+"_"+node+".metrics")).write_bytes(b)
 v=re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M);assert v and all(float(x)==0for x in v)
atomic_json(r/(label+".json"),dict(at=utc(),roots=out,NPU32_same=True,idle=True));print("joint32 bothphysicalhosts andoneAPI epoch verified")
