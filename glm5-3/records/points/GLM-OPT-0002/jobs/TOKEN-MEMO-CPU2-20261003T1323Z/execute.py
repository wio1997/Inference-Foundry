from pathlib import Path
import json,subprocess,sys,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0178";roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
s=json.loads((r/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def probe(label):
 out={}
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=100);(j/(label+"_"+key+".stdout")).write_bytes(z.stdout);(j/(label+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
  expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 for node,port in[("166",9081),("167",9900)]:
  b=http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=15).read();(j/(label+"_"+node+".metrics")).write_bytes(b);d={}
  for l in b.decode().splitlines():
   if not l or l.startswith("#"):continue
   k=l.split("{")[0].split()[0]
   if k in ["vllm:request_success_total","vllm:generation_tokens_total","vllm:prompt_tokens_total","vllm:prefix_cache_queries_total","vllm:prefix_cache_hits_total","vllm:num_preemptions_total","vllm:num_requests_running","vllm:num_requests_waiting"]:d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
  assert d["vllm:num_requests_running"]==d["vllm:num_requests_waiting"]==0;out[node]=d
 return out
before=probe("before")
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(j/"tokens.py")
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=300);(j/"client.stdout").write_bytes(z.stdout);(j/"client.stderr").write_bytes(z.stderr);z.check_returncode()
acks=[json.loads(l)for l in z.stdout.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
after=probe("after");assert after==before
out=json.loads((j/"token_summary.json").read_text());out.update(at=utc(),native32_same=True,native_counter_deltas0=True,actual_client_SDK_init_finalize=[acks[0],acks[-1]],native_metrics=after)
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="CPU memo mock contracts and real full152 token equality/counts verified; native32 unchanged/inferencecounter0/SDKinit-final0; actualHTTPChat E2E pending.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="CPU source/API/tokenization/cost/native32/counter0")],unknowns=out["limitations"],decision_request=None,next_check_at=None));print(json.dumps(dict(valid=True,evidence=e)))
