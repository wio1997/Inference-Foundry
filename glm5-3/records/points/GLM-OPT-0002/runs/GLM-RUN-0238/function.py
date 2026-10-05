from pathlib import Path
import json,sys,subprocess,shlex,re,urllib.request,hashlib
r=Path(__file__).parent
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
from owner_guard import start_watchdog,guard
from native_client_execution import run_native_client
import controls
start_watchdog();controls.verify("compat_before")
native=["python3","-u","/data/tiankuan/wio/glm52-pd/deploy/plugins/local_engines137/native_acl_lifecycle.py","script",str(r/"tools_client.py")]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(r)+":/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime:$PYTHONPATH; export GLM_TOOL_REPLICA=D1_compat GLM_TOOL_URL=http://127.0.0.1:8000; export GLM_TOOL_METRICS_URLS="+shlex.quote(json.dumps(dict(D0="http://172.16.10.166:9081",D1="http://172.16.10.167:9900")))+"; exec "+shlex.join(native)
exe=run_native_client(["docker","exec","glm52-single","bash","-c",shell],record_path=r/"native_tools_execution.json",stdout_path=r/"native_tools.stdout",stderr_path=r/"native_tools.stderr",expected_native_argv=native,timeout_s=1000,guard=guard)
assert exe["status"]=="succeeded"and not exe["native_client_alive"]and not exe["signal_attempts"]
events=[json.loads(l)for l in(r/"native_tools.stdout").read_text().splitlines()if l.startswith('{"event":')];assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
tools=json.loads((r/"tools_D1_compat/summary.json").read_text());assert tools["valid"]and len(tools["requests"])==4and tools["cleanup"]["native_idle"]
controls.verify("compat_after")
def metrics(f):
 d={}
 for l in f.read_text().splitlines():
  if l.startswith("vllm:"):
   k=l.split("{")[0].split()[0];d[k]=d.get(k,0)+float(l.rsplit(" ",1)[1])
 return d
delta={}
for key,node in[("D0","node0"),("D1","node1")]:
 a=metrics(r/("compat_before_"+node+".metrics"));b=metrics(r/("compat_after_"+node+".metrics"));delta[key]={k:b[k]-a[k]for k in a if k.endswith("_total")}
assert all(x==0for x in delta["D0"].values())and delta["D1"]["vllm:generation_tokens_total"]==tools["effective_output_tokens"]and delta["D1"]["vllm:request_success_total"]==4and delta["D1"]["vllm:num_preemptions_total"]==0
trace=[json.loads(l)for l in(r.parent/"GLM-RUN-0125/router_trace.jsonl").read_text().splitlines()]
for row in tools["requests"]:
 head=r.name+"-tool-D1_compat-"+row["name"];leases=[x for x in trace if x["event"]=="lease_acquired"and x.get("request_header_id")==head];assert len(leases)==1and leases[0]["replica"]=="D1"
 wires=[x for x in trace if x["event"]=="upstream_stream_contract"and x.get("lease_id")==leases[0]["lease_id"]];releases=[x for x in trace if x["event"]=="lease_released"and x.get("lease_id")==leases[0]["lease_id"]];assert len(wires)==len(releases)==1and releases[0]["released"]and not releases[0]["backend_failure"]and Path(wires[0]["wire_path"]).read_bytes()==Path(row["wire"]["path"]).read_bytes()
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=15)as x:state=json.loads(x.read())
by={x["id"]:x for x in state["replicas"]};assert by["D0"]["group_faulted"]and not by["D1"]["group_faulted"]and all(not x["active_requests"]for x in by.values())
atomic_json(r/"compatibility_summary.json",dict(at=utc(),functional_acceptance=True,actual_completed_requests=4,effective_output_tokens=tools["effective_output_tokens"],native_owner="D1",native_delta=delta,tools=tools,SDKinit_finalize0=True,native32_same=True,public228_retained=True,D0_fault_preserved=True,native_operator_changes=0,models_started=0,models_signalled=0,public_restarts=0,policywrites=0,full_native_request_equivalence=False,Current=None,limits=["FreshnativeV1/D1 TP4PP4DCP4K1 toolauto-required-named-none wire/tokenIDs/semantic/usage valid; D0V2required236FAILED/compatroute notyetimplemented","NativeD0fault persisted/no clear/no replay/modelops; onlyhealthyD1 realrequests","No thinkingbudget/fullResponses139/KEEP/stablecapacity proof; conditionalcap2mixed benefitstillSLAFAIL"]))
print(json.dumps(dict(valid=True,outputs=tools["effective_output_tokens"],completed=4,owner="D1")))
