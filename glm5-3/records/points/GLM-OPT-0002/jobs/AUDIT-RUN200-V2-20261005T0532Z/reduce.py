
from pathlib import Path
import sys,json,hashlib,subprocess,shlex,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from native_engines_service_config import checked_config
from native_identity_observer import observe,owner_alive
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;b=j.parents[1];r=b/"runs/GLM-RUN-0200";folder=r/"restored"
def ref(p):
 raw=Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
s=json.loads((r/"state.json").read_text());assert s["status"]=="completed"and not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text());assert hashlib.sha256((r/"controller_spec.json").read_bytes()).hexdigest()==s["spec_sha256"]
for st in spec["stages"]:
 assert json.loads((r/(st["id"]+".phase.json")).read_text())["status"]=="succeeded"
 for p in st["sources"]:assert Path(p["path"]).read_bytes()==Path(p["snapshot"]).read_bytes()and ref(p["path"])["sha256"]==p["sha256"]
config,state=checked_config(folder/"service_config.json");assert config["placement"]==dict(kind="shape_split",input_threshold_bytes=32768,prefill_members=["D1"],decode_members=["D0"])
assert len(config["native_domains"])==2 and all(len(v["members"])==1 for v in config["native_domains"])
roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text());plans=json.loads((folder/"standalone_launch.json").read_text())
physical={};operator_markers=[];rankmaps={}
for key,o in roots.items():
 node=o["host"];K=3 if node=="166"else 1
 args=["python3","-c",(r/"live_probe.py").read_text()]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout)
 expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
 assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
 a=o["argv"];assert "--headless"not in a
 for flag,value in[("--nnodes","1"),("--tensor-parallel-size","4"),("--pipeline-parallel-size","4"),("--decode-context-parallel-size","4"),("--max-num-batched-tokens","8192")]:assert a[a.index(flag)+1]==value
 assert json.loads(a[a.index("--speculative-config")+1])["num_speculative_tokens"]==K
 assert v["env"]["VLLM_PP_LAYER_PARTITION"]=="22,20,20,16"
 code="from pathlib import Path;import json,hashlib;p=Path('/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp200');print(json.dumps(dict(policy=json.loads((p/'issue_budget_policy.json').read_text()),sha=hashlib.sha256((p/'issue_budget_scheduler_v5.py').read_bytes()).hexdigest())))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,capture_output=True,timeout=60);z.check_returncode();policy=json.loads(z.stdout)
 assert policy["policy"]==dict(schema_version=1,cohort_id="GLM-COHORT-0200",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 assert policy["sha"]==ref(b.parents[2]/"runtime/issue_budget_scheduler_v5.py")["sha256"]
 physical[node]=dict(root_same=True,NPU16_owned=True,K=K,proof=ref(j/(key+".owner.stdout")),policy=policy)
 log=(r/("native_resident_"+key+".stdout")).read_text()
 markers=[json.loads(l.split("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED ",1)[1])for l in log.splitlines()if "GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "in l]
 primary=json.loads((b/"runs/GLM-RUN-0190/restored/PP_empty_guard_workers.json").read_text())["markers"][0]
 assert len(markers)==16and all(x==primary for x in markers);operator_markers+=markers
 # frozen native init rank logs prove local16 and PP4TP4 groupgeometry, not percollective device activity
 assert "Graph capturing finished"in log
 ranklines=[l for l in log.splitlines()if "world_size=16 rank="in l or "in world size 16 is assigned"in l]
 rankmaps[node]=dict(lines=ranklines,source=ref(r/("native_resident_"+key+".stdout")),limits="native init labels; not physical percollective trace")

 rank_entries=[]
 for line in ranklines:
  m=re.search(r"\(Worker pid=(\d+)\).*world_size=16 rank=(\d+) local_rank=(\d+)",line)
  if m:rank_entries.append(dict(container_pid=int(m[1]),rank=int(m[2]),local_rank=int(m[3])))
 assert len(rank_entries)==16 and {v["rank"]for v in rank_entries}==set(range(16))and all(v["rank"]==v["local_rank"]for v in rank_entries)
 mapcode="import pathlib,json,sys;rows=[];a=json.load(sys.stdin);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()\nfor x in a:\n p=pathlib.Path('/proc/'+str(x['pid']));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();assert v[0]not in['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity'];ns=next(l for l in(p/'status').read_text().splitlines()if l.startswith('NSpid:')).split()[1:];rows.append(dict(host_pid=x['pid'],identity=x['identity'],container_pid=int(ns[-1])))\nprint(json.dumps(rows))"
 args=["python3","-c",mapcode]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 targets=[x for x in members[key]["owned_targets"]if x["pid"]in members[key]["npu_worker_pids"]]
 z=subprocess.run(args,input=json.dumps(targets).encode(),capture_output=True,timeout=60);z.check_returncode();(j/(key+".NSpid.stdout")).write_bytes(z.stdout);nspids=json.loads(z.stdout)
 assert {v["container_pid"]for v in nspids}=={v["container_pid"]for v in rank_entries}
 mapped=[]
 for v in rank_entries:
  hostrow=next(x for x in nspids if x["container_pid"]==v["container_pid"])
  expectedPP=v["rank"]//4;expectedTP=v["rank"]%4
  label="Worker_PP"+str(expectedPP)+"_TP"+str(expectedTP)+"_DCP"+str(expectedTP)+" pid="+str(v["container_pid"])
  assert label in log
  mapped.append(dict(v,**hostrow,PP=expectedPP,TP=expectedTP,DCP=expectedTP))
 rankmaps[node]["actual_rank_HOST_NPU_map"]=mapped
 rankmaps[node]["NSpid_proof"]=ref(j/(key+".NSpid.stdout"))

for node in["166","167"]:
 raw=(r/("CPU_exactplan_"+node+".stdout")).read_text();ev=[json.loads(l)for l in raw.splitlines()if l.startswith('{"event":')]
 assert ev[0]["event"]=="task_acl_init"and ev[-1]["event"]=="task_acl_finalize"and ev[0]["returncode"]==ev[-1]["returncode"]==0
 full=next(x for x in ev if x["event"]=="fullCLI_PP32_exactplan");assert full["TP"]==full["DCP"]==4and full["world"]==16and full["workers"]==full["models"]==full["inference"]==0
# Verify exact oldnative target receipts, oldSTORE prelease rejection and oldpublic SDKfinal0
oldroots=json.loads((r/"standalone_root_identities.json").read_text());oldmembers=json.loads((r/"standalone_native_members.json").read_text());signals=[]
for key,o in oldroots.items():
 fresh=json.loads((r/("stop_fresh_"+key+".stdout")).read_text());known={x["pid"]:x["identity"]for x in fresh["owned_targets"]}
 assert fresh["root"]==o and fresh["npu_worker_pids"]==oldmembers[key]["npu_worker_pids"]
 events=[json.loads(l)for l in(r/("stop_"+key+".stdout")).read_text().splitlines()if l.startswith("{")]
 assert events[-1]["event"]=="cleanup_complete"and events[-1]["signals_to_unknown"]==0 and events[-1]["npu_workers"]==0
 for x in events:
  if x["event"]=="signal_attempt":assert x["pid"]in known and x["identity"]==known[x["pid"]];signals.append(x)
fault=json.loads((r/"physical_fault_ack.json").read_text());assert fault["both_exact_old_native_domains_retired"]and all(x["status"]=="fault"for x in fault["observer"]["groups"])
retired=json.loads((r/"client_retired_summary.json").read_text());assert retired["valid"]and retired["effective_output_tokens"]==0and all(x["status"]==503and x["rejected_before_lease_and_RPC"]for x in retired["requests"])
retire=json.loads((r/"retire196_public.json").read_text());assert retire["SDKinit_finalize0"]and not same_process(retire["exact_owned_host"])
assert (b/"runs/GLM-RUN-0196/restored/identity_observer/terminal.json").exists()
# Complete pilot IDs and native counters
pilot=json.loads((r/"pilot_SDK_summary.json").read_text());assert pilot["measurement_valid"]and pilot["SDKinit_finalize0"]and pilot["output_tokens"]==192
pilot_rows=[]
for i in[0,1]:
 d=r/("pilot_"+str(i));v=json.loads((d/"pilot_summary.json").read_text());assert v["functional_acceptance"]and v["effective_public_output_tokens"]==96
 for x in v["requests"]:
  wire=Path(x["wire"]["path"]);assert ref(wire)==x["wire"];raw=wire.read_bytes();obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True);obs.feed(raw)
  assert obs.done and not obs.contract_error and not obs.contract_unknown and obs.finish_reasons=={"0":"length"}and obs.contract()["usage"]==x["usage"]
  ids=[]
  for frame in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
   data=b"\n".join(l[5:].lstrip(b" ")for l in frame.splitlines()if l.startswith(b"data:"))
   if not data or data==b"[DONE]":continue
   event=json.loads(data);assert not event.get("error")
   for choice in event.get("choices",[]):assert choice["index"]==0;ids+=choice.get("token_ids")or[]
  assert len(ids)==x["usage"]["completion_tokens"]and all(type(z)is int and z>=0for z in ids)
  assert x["usage"]["prompt_tokens"]==(21 if x["name"].startswith("short")else 81932)
  pilot_rows.append(dict(owner=x["owner"],name=x["name"],usage=x["usage"],wall_s=x["wall_s"],ttft_s=x["ttft_s"],wire=x["wire"],committed_ID_count=len(ids),committed_ID_sha256=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()))
final=json.loads((r/"client_final_summary.json").read_text());assert final["valid"]and final["effective_output_tokens"]>0
for x in final["requests"]:
 assert ref(x["body"]["path"])==x["body"]and ref(x["wire"]["path"])==x["wire"]
 if x["status"]==503:assert x["rejected_before_lease_and_RPC"];continue
 assert x["native_owner"]in["D0","D1"]and x["native_wire"]["sha256"]==x["wire"]["sha256"]and Path(x["native_wire"]["path"]).read_bytes()==Path(x["wire"]["path"]).read_bytes()
 if x.get("effective_output_tokens"):assert x["usage"].get("completion_tokens",x["usage"].get("output_tokens"))==x["effective_output_tokens"]
creates={key:next(x for x in final["requests"]if x["name"]==key+"_create")for key in["D0","D1"]}
for key,row in creates.items():
 assert row["native_owner"]==key and row["effective_output_tokens"]==32
 children=[x for x in final["requests"]if x["name"]==key+"_previous"];assert len(children)==1and children[0]["native_owner"]==key and children[0]["affinity_applied"]and children[0]["effective_output_tokens"]==16
 for name in[key+"_retrieve",key+"_child_retrieve"]:assert next(x for x in final["requests"]if x["name"]==name)["affinity_applied"]
for name in["client_before","client_retired","client_final"]:
 ev=[json.loads(l)for l in(r/(name+".stdout")).read_text().splitlines()if l.startswith('{"event":')]
 assert ev[0]["event"]=="task_acl_init"and ev[-1]["event"]=="task_acl_finalize"and ev[0]["returncode"]==ev[-1]["returncode"]==0

def totals(path):
 out={}
 for l in Path(path).read_text().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{",1)[0].split()[0];out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
native_end={}
for node,key in[("166","D0"),("167","D1")]:
 data=totals(r/("client_final_after_"+key+".metrics"))
 mine=[v for v in final["requests"]if v.get("native_owner")==key and v.get("effective_output_tokens")]
 out=sum(v["effective_output_tokens"]for v in mine)
 assert data["vllm:generation_tokens_total"]==96+out and data["vllm:request_success_total"]==7and data["vllm:num_preemptions_total"]==0
 assert data["vllm:num_requests_running"]==data["vllm:num_requests_waiting"]==data["vllm:kv_cache_usage_perc"]==0
 assert data["vllm:prompt_tokens_total"]==81953+sum(v["usage"].get("prompt_tokens",v["usage"].get("input_tokens"))for v in mine)
 native_end[key]={k:v for k,v in data.items()if k.endswith("_total")or k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]}
assert sum(v["vllm:generation_tokens_total"]for v in native_end.values())==192+final["effective_output_tokens"]

public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])
assert [v.decode()for v in Path("/proc/"+str(public["host"]["pid"])+"/cmdline").read_bytes().split(bytes([0]))if v]==public["argv"]
assert ref(public["container"]["config"]["path"])==public["container"]["config"]and public["container"]["native_domains"]==config["native_domains"]
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as res:placement=json.loads(res.read())
assert {x["id"]for x in placement["replicas"]}=={"D0","D1"}and all(not x["active_requests"]and not x["group_faulted"]and not x["draining"]and x["placement_policy"]=="shape_split"for x in placement["replicas"])
observed=observe(folder/"service_config.json");assert len(observed["groups"])==2and all(x["status"]=="healthy"for x in observed["groups"])
assert owner_alive(json.loads((folder/"identity_observer/process_owner.json").read_text()))
atomic_json(j/"public_service_proof.json",dict(public,at=utc(),placement=placement))
limits=["Functional finitefit/API/STORE proof not capacityKEEP or repeatedcausal gain; CurrentNone/stablecapacity/globalboundunknown","Newepochs/independent16geometry vs197198joint32 and192twoindependent16differentpublicrouting; K3/K1 retained, D1V5c1 vs oldV3c1 delegatesnative unchanged but notisolatedcausalestimate","199precontroller sourcepinfixtureINVALID preserved/noGPU replay; 200fullCPU/rawoldretirement andfit/function audited","Rankinit logs/source-worker labels not percollective device trace; shape_split32768 HTTPinputbytes routes smallD0-largeD1, changesloadallocation notrequestwork","Existing139147149-151168175179184 fullAPI/tools/cancel/drain/nativeerrors evidence reusedriskmatched; no handchecklist shrinkscompleteframeworkfunction"]
out=dict(at=utc(),run_id=r.name,measurement_valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",Current=None,source_count=len(spec["stages"][0]["sources"]),effective_outputs=192+final["effective_output_tokens"],native_complete_requests=14,native32_same_owned_idle=True,physical=physical,native_domains=config["native_domains"],placement=config["placement"],pilot=pilot,pilot_ID_proof=pilot_rows,native_end=native_end,final_client=ref(r/"client_final_summary.json"),retired196STORE503=ref(r/"client_retired_summary.json"),oldnative_exactsignals=signals,public=ref(j/"public_service_proof.json"),operator_guard_primary_same=True,rankinit=rankmaps,private_c1serial1_both=True,helpers=0,STORE_replication=False,limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="200 independent16x2 shape32768 functional VALID/fullnativeIDs-SDK-STORE both/oldjoint196retired; capacityINCONCLUSIVE",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="source/ranks/native32/pilotIDs/clientSDK/STORE-affinity-retirement/publicshape")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,outputs=out["effective_outputs"],public=out["public"],reduction=ref(j/"reduction.json"))))
