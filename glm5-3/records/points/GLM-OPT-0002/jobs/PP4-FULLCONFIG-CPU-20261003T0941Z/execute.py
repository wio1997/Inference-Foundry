from pathlib import Path
import sys,json,subprocess,hashlib,re,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0165"
state=json.loads((r/"state.json").read_text());assert state["status"]=="completed"and not same_process(state["owner"])
http=__import__("urllib.request",fromlist=[""]).build_opener(__import__("urllib.request",fromlist=[""]).ProxyHandler({}))
def ref(f):
 b=Path(f).read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text())
def probe(label):
 out={}
 for key,o in roots.items():
  args=["python3","-c",(r/"live_probe.py").read_text()]
  if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
  z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(label+"_"+key+".stdout")).write_bytes(z.stdout);(j/(label+"_"+key+".stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
  expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]};assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 for node,port in[("166",9081),("167",9900)]:
  b=http.open("http://172.16.10."+node+":"+str(port)+"/metrics",timeout=15).read();(j/(label+"_"+node+".metrics")).write_bytes(b);d={}
  for l in b.decode().splitlines():
   if l and not l.startswith("#"):
    k=l.split("{")[0].split()[0]
    if k in["vllm:request_success_total","vllm:generation_tokens_total","vllm:prompt_tokens_total","vllm:prefix_cache_queries_total","vllm:prefix_cache_hits_total","vllm:num_preemptions_total","vllm:num_requests_running","vllm:num_requests_waiting"]:d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
  assert d["vllm:num_requests_running"]==d["vllm:num_requests_waiting"]==0;out[node]=d
 return out
before=probe("before")
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH=/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137:/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; python3 /data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py script "+str(j/"cpu_probe.py")+" PP4_22_20_20_16"
z=subprocess.run(["docker","exec","glm52-single","bash","-c",shell],capture_output=True,timeout=240);(j/"cpu.stdout").write_bytes(z.stdout);(j/"cpu.stderr").write_bytes(z.stderr);z.check_returncode()
lines=z.stdout.decode().splitlines();events=[json.loads(l)for l in lines if l.startswith('{"event":')]
acks=[x for x in events if x["event"].startswith("task_acl_")];assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
cpu=next(x for x in events if x["event"]=="CPU_geometry_config");assert cpu["config_accepted"]and cpu["native_use_eagle"]and cpu["effective_DCPblock_size"]==512and cpu["NPU_tensor_allocations"]==cpu["inference"]==cpu["model_instances"]==0
d=p/"jobs/TOKEN-INPUT-CPU3-20261003T0837Z";ids={};refs={}
for name in["warmup0","full0","full1"]:
 f=d/(name+"_ids.json");b=f.read_bytes();v=json.loads(b);assert isinstance(v,list);ids[name]=v;refs[name]=ref(f)
def lcp(a,b):
 for i,(x,y)in enumerate(zip(a,b)):
  if x!=y:return i
 return min(len(a),len(b))
prefix={name:lcp(ids["warmup0"],ids[name])for name in["full0","full1"]}
assert prefix==dict(full0=73738,full1=73738)
after=probe("after");assert before==after
out=dict(at=utc(),valid=True,kind="native_TP4PP4DCP4_fullconfig_CPU",SDKinit_finalize0=True,native32_same=True,native_inference_counter_delta0=True,native_config=cpu,actual_native_token_inputs=refs,warm_full_common_prefix=prefix,observed_prior165_hit_tokens_per_full=72704,conditional_explanation="PP4TP4DCP4 fullnativeconfig CPUaccepted/boundaries22,42,62full/native512schedulerblock; expectedMTPmetadata margin512 vs1024 only, futureactualweights-HCCL-Graph-fit-quality-perfunknown. Native64coerced128, backend128 only, no guardbypass.",block_edits=0,model_worker_tensor_operations=0,limits=["Removingdrop could violate MTP hiddenstate dependency; no cacheguard bypass or claim safe","Changingblock/hash/alignment affects nativeKVlayout and guards; requires config/source feasibility and actualE2E before benefit claim","Cannotattribute remaining4.4s prefill or globalcapacity from CPUfunction alone"],native_metrics=after)
atomic_json(j/"reduction.json",out);e=ref(j/"reduction.json")
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="NativeTP4PP4DCP4/22,20,20,16 fullCPUconfig/allHFboundaries accepted/SDK0/no models-workers-tensors-inference/native32same; actualweights-HCCL-Graph-fit-quality-perfunknown.",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**e,locator="Actualinstallednativefunction/syntheticmetadata/config-SDK/actualtokenLCP/zeroinference")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(out))
