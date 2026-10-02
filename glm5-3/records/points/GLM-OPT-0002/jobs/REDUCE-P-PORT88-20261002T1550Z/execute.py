from pathlib import Path
import json,sys,ast,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0088";s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and not same_process(s["owner"])
owners=json.loads((r/"startup_model_identities.json").read_text());tree=ast.parse((r/"deploy.py").read_text());cs=[n.value.value for n in ast.walk(tree)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str)and"npu_worker_pids"in n.value.value and"API_alive"in n.value.value];assert len(cs)==2 and cs[0]==cs[1];code=cs[0];members={};logs={};policy=json.loads((r/"issue_budget_policy.json").read_text());opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for key,o in owners.items():
 args=["python3","-c",code]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(o).encode(),capture_output=True,timeout=90);(j/(key+"_scope.stdout")).write_bytes(z.stdout);(j/(key+"_scope.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout);assert v["API_alive"]and v["API_owner"]==o and len(v["npu_worker_pids"])==16 and v["env"]["GLM_ISSUE_BUDGET_COHORT"]=="GLM-COHORT-0088";members[key]=v
 x=next(x for x in json.loads((r/"planned_launch.json").read_text())if x["node"]==o["host"]);args=["cat",x["log"]]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=30);z.check_returncode();f=j/("native_"+o["host"]+".log");f.write_bytes(z.stdout);logs[key]=dict(path=str(f),bytes=len(z.stdout),sha256=hashlib.sha256(z.stdout).hexdigest(),text=z.stdout.decode(errors="replace"))
 if key=="D1":
  with opener.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as res:b=res.read();assert res.status==200
  (j/"D1_idle.metrics").write_bytes(b);vals=[float(v)for v in re.findall(r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M)];assert len(vals)>=2 and all(v==0 for v in vals)
  assert logs[key]["text"].count("Graph capturing finished")>=16
P=logs["D0"]["text"];errors=[x for x in P.splitlines()if"KVCacheSendingThread encountered exception"in x];assert len(errors)==2 and all("Address already in use"in x for x in errors)and any(":36407"in x for x in errors)and any(":36415"in x for x in errors)
raw=subprocess.check_output(["ss","-ltnp"],text=True);(j/"P_sockets.txt").write_text(raw);clashes=[x for x in raw.splitlines()if re.search(r":364(?:07|15)\s",x)];assert len(clashes)==2
pids=[int(v)for l in clashes for v in re.findall(r"pid=(\d+)",l)];assert set(pids)<=set(members["D0"]["npu_worker_pids"])
ns={x["pid"]:x["comm"]for x in members["D0"]["owned_targets"]};port_rows=[dict(port=int(re.search(r":(364(?:07|15))\s",l).group(1)),pid=int(re.search(r"pid=(\d+)",l).group(1)),comm=ns[int(re.search(r"pid=(\d+)",l).group(1))],line=l)for l in clashes]
assert any(x["port"]==36407 and"TP10_"in x["comm"]for x in port_rows)and any(x["port"]==36415 and"TP12_"in x["comm"]for x in port_rows)
assert not any((r/x).exists()for x in["semantic_summary.json","pilot_summary.json","attempts.json"])
cpu={}
for node in["166","167"]:
 f=r/("CPU_fullCLI_"+node+".stdout");events=[json.loads(l)for l in f.read_text().splitlines()if l.startswith('{"event":')];assert len(events)==3 and events[0]["returncode"]==events[-1]["returncode"]==0 and events[1]["event"]=="full_native_CLI_API_config_valid";cpu[node]=events
atomic_json(r/"post_failure_native_member_identities.json",members)
out=dict(at=utc(),verdict="INVALID",functional_acceptance=False,effective_public_output_tokens=0,new_inference_requests=0,controller_terminal=s,P_startup_failure="ZMQ nativeMooncake sendingthreads bind36407/36415 collidedwithotherownednativeworker ephemeral listeners",error_lines=errors,collision_sockets=port_rows,ephemeral_range=Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().strip(),P_API_live_unready=True,P_NPU16_alive=True,D_API_ready_idle=True,D_NPU16_same=True,D_nativeGraph_complete=True,CPU2=cpu,owned_resources=ref if False else dict(path=str(r/"post_failure_native_member_identities.json"),bytes=(r/"post_failure_native_member_identities.json").stat().st_size,sha256=hashlib.sha256((r/"post_failure_native_member_identities.json").read_bytes()).hexdigest()),native_logs={k:{f:v[f]for f in["path","bytes","sha256"]}for k,v in logs.items()},limits=["No semantic/pilot requeststarted; nativeport allocation conflict notcapacity/modelmath rejection","36400..36415 liesin32768..60999 OS ephemeralrange; exactcollidingownersTP10/TP12verified taskancestry, lowerunusednewrangecandidateaddressesclass; no kernel/globalOS/network change","D88canretainownedhealthyAPI/NPU16 whileonlyP rerunsstartup; modelmemory3GiBKV/capturefit evidenceD only, fullP fit/PD unresolved"])
atomic_json(r/"reduction_brief.json",out);f=r/"reduction_brief.json";b=f.read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="88INVALID PZMQ36407/36415conflictwithownedTP10/TP12ephemeralports; nativeD88Graphreadyidle/NPU16 retained, PexactAPI/NPU16liveunready; no requests",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="audit",path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativeerror/sockets/portowners/NPU32/APIidentity/CPU2SDK0/DGraphidle")],unknowns=["PGraph/semantics/PD/performance unexecuted"],decision_request=None,next_check_at=None));print(json.dumps(out))
