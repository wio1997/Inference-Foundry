from common import *
from native_client_execution import run_native_client
m=json.loads((r/"restored/native_engines_resident.json").read_text())
def verify(label):
 for e in m["engines"]:
  for key,o in e["roots"].items():fresh(o,e["members"][key],label+"_"+e["replica_id"]);native_idle(o["host"],9081 if o["host"]=="166"else 9900,label)
verify("feature_before")
native=["python3","-u",str(plug/"native_acl_lifecycle.py"),"script",str(r/"tools_client.py")]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(bundle)+":"+str(r)+":$PYTHONPATH; export GLM_TOOL_REPLICA=V14_mixed GLM_TOOL_URL=http://127.0.0.1:8000 GLM_TOOL_METRICS_URLS="+shlex.quote(json.dumps(dict(D0="http://172.16.10.166:9081",D1="http://172.16.10.167:9900")))+"; exec "+shlex.join(native)
exe=run_native_client(["docker","exec","glm52-single","bash","-c",shell],record_path=r/"tools_execution.json",stdout_path=r/"tools.stdout",stderr_path=r/"tools.stderr",expected_native_argv=native,timeout_s=1000,guard=guard)
assert exe["status"]=="succeeded"and not exe["native_client_alive"]and not exe["signal_attempts"]
acks=[json.loads(l)for l in(r/"tools.stdout").read_text().splitlines()if l.startswith('{"event":')];assert [a["event"]for a in acks]==["task_acl_init","task_acl_finalize"]and all(a["returncode"]==0for a in acks)
tools=json.loads((r/"tools_V14_mixed/summary.json").read_text());assert tools["valid"]and len(tools["requests"])==4and tools["cleanup"]["native_idle"]
verify("feature_after")
trace=[json.loads(l)for l in(state/"router_trace.jsonl").read_text().splitlines()]
expected={"auto":"D0","required":"D1","named":"D1","none":"D0"};credits={k:dict(outputs=0,prompts=0,completed=0)for k in ["D0","D1"]}
for row in tools["requests"]:
 head=r.name+"-tool-V14_mixed-"+row["name"];leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==head];assert len(leases)==1and leases[0]["replica"]==expected[row["name"]]
 lease=leases[0];wires=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==lease["lease_id"]];releases=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==lease["lease_id"]]
 assert len(wires)==len(releases)==1and releases[0]["released"]and not releases[0]["backend_failure"]and Path(wires[0]["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes()and lease["body_sha256"]==ref(r/"tools_V14_mixed"/(row["name"]+".body.json"))["sha256"]
 row["native_owner"]=lease["replica"];row["lease_id"]=lease["lease_id"];c=credits[lease["replica"]];c["outputs"]+=row["usage"]["completion_tokens"];c["prompts"]+=row["usage"]["prompt_tokens"];c["completed"]+=1
def metrics(f):
 out={}
 for l in f.read_text().splitlines():
  if l.startswith("vllm:"):
   k=l.split("{")[0].split()[0];out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
delta={}
for key,node in [("D0","166"),("D1","167")]:
 a=metrics(r/("feature_before_"+node+".metrics"));b=metrics(r/("feature_after_"+node+".metrics"));delta[key]={k:b[k]-a[k]for k in a if k.endswith("_total")}
 assert delta[key]["vllm:generation_tokens_total"]==credits[key]["outputs"]and delta[key]["vllm:prompt_tokens_total"]==credits[key]["prompts"]and delta[key]["vllm:request_success_total"]==credits[key]["completed"]and delta[key]["vllm:num_preemptions_total"]==0
snap=request("/control/replicas");assert len(snap["replicas"])==2and all(not x["group_faulted"]and not x["active_requests"]for x in snap["replicas"])
with http.open("http://127.0.0.1:8000/v1/responses/resp_glm_run211_D1_new",timeout=15)as response:raw=response.read();assert response.status==200
assert json.loads(raw)==json.loads((r/"before_STORE211.wire").read_text());(r/"after_STORE211.wire").write_bytes(raw)
atomic_json(r/"feature_summary.json",dict(at=utc(),functional_acceptance=True,completed=4,effective_output_tokens=tools["effective_output_tokens"],requests=tools["requests"],native_credits=credits,native_delta=delta,native32_same=True,native_epochs_unchanged=True,SDKinit_finalize0=True,public_V14=True,STORE211_retained=True,Current=None,limits=["Four fresh complete real tools prove V14 required/named selects V1 while ordinary auto/none retains V2","No native schema/thinking budget/Responses lifecycle E2E yet; no performance KEEP or stable capacity upper bound"]))
print(json.dumps(dict(valid=True,completed=4,outputs=tools["effective_output_tokens"],routing=expected)))
