import asyncio,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
sys.path.insert(0,sys.argv[1])
from native_engines_service_config import compile_config,checked_config
from response_affinity_gateway_v11 import create_app
j=Path(sys.argv[1]);runtime=Path("/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
argv=json.loads((j/"native_CPU_argv.json").read_text())
engines=[]
for index,host in enumerate(["166","167"]):
 a=argv[:]
 for option,value in[("--tensor-parallel-size","8"),("--decode-context-parallel-size","8"),("--nnodes","1"),("--node-rank","0"),("--port","9081"if host=="166"else"9900"),("--master-addr","172.16.10."+host)]:
  a[a.index(option)+1]=value
 if "--headless"in a:a.remove("--headless")
 plan=dict(host=host,node_rank=0,api=True,argv=a,environment=dict(VLLM_PP_LAYER_PARTITION="42,36",fixture=True))
 root=dict(host=host,pid=1000+index,argv=a,role="API",identity=dict(boot_id="CPU-fixture-"+host,start_ticks="100"))
 targets=[dict(pid=2000+i,identity=dict(boot_id=root["identity"]["boot_id"],start_ticks=str(200+i)))for i in range(16)]
 member=dict(root=root,root_alive=True,npu_worker_pids=[x["pid"]for x in targets],owned_targets=targets)
 engines.append(dict(id="local-"+host,replica_id="D"+str(index),plans=dict(node0=plan),roots=dict(node0=root),members=dict(node0=member)))
material=dict(engines=engines,placement=dict(kind="shape_split",input_threshold_bytes=8192,prefill_members=["D1"],decode_members=["D0"]))
state=j/"fixture_state";config=compile_config(material,state);assert len(config["native_domains"])==2and all(g["world_size"]==16and g["nnodes"]==1for g in config["engine_geometry"].values())
contracts=["two_independent_local_engine_domains"]
fixture=j/"resident_fixture";fixture.mkdir();(fixture/"native_engines_resident.json").write_text(json.dumps(material,indent=2)+"\n")
output=j/"rendered_config.json"
z=subprocess.run([sys.executable,str(j/"native_engines_service_config.py"),"--resident-dir",str(fixture),"--state-dir",str(state),"--output",str(output)],capture_output=True);z.check_returncode();(j/"render.stdout").write_bytes(z.stdout);(j/"render.stderr").write_bytes(z.stderr);assert checked_config(output)[0]["native_domains"]==config["native_domains"];contracts.append("actual_render_and_checked_provenance")
for kind in["duplicate_engine_ID","duplicate_replica_ID","duplicate_host","native_root_role","missing_NPU","wrong_worker_boot","KV_connector","invalid_placement","duplicate_shape_member","uncovered_shape_member","bool_shape_threshold"]:
 bad=copy.deepcopy(material)
 if kind=="duplicate_engine_ID":bad["engines"][1]["id"]=bad["engines"][0]["id"]
 elif kind=="duplicate_replica_ID":bad["engines"][1]["replica_id"]=bad["engines"][0]["replica_id"]
 elif kind=="duplicate_host":
  e=copy.deepcopy(bad["engines"][0]);e["id"]="same-host-extra";e["replica_id"]="D1";bad["engines"][1]=e
 elif kind=="native_root_role":bad["engines"][1]["roots"]["node0"]["role"]="headless"
 elif kind=="missing_NPU":bad["engines"][1]["members"]["node0"]["npu_worker_pids"].pop()
 elif kind=="wrong_worker_boot":bad["engines"][1]["members"]["node0"]["owned_targets"][0]["identity"]["boot_id"]="wrong-boot"
 elif kind=="KV_connector":
  e=bad["engines"][1];e["plans"]["node0"]["argv"]+=["--kv-transfer-config","{}"];e["roots"]["node0"]["argv"]=e["plans"]["node0"]["argv"];e["members"]["node0"]["root"]=e["roots"]["node0"]
 elif kind=="invalid_placement":bad["placement"]["kind"]="unknown"
 elif kind=="duplicate_shape_member":bad["placement"]["decode_members"]=["D0","D0"]
 elif kind=="uncovered_shape_member":bad["placement"]["prefill_members"]=["missing"]
 elif kind=="bool_shape_threshold":bad["placement"]["input_threshold_bytes"]=True
 try:compile_config(bad,state)
 except ValueError:contracts.append(kind+"_rejected")
 else:raise AssertionError(kind+"accepted")
changed=copy.deepcopy(material);changed["engines"][1]["members"]["node0"]["owned_targets"][0]["identity"]["start_ticks"]="999"
new=compile_config(changed,state);assert new["native_domains"][0]==config["native_domains"][0]and new["native_domains"][1]["epoch"]!=config["native_domains"][1]["epoch"];contracts.append("only_changed_engine_epoch_retired")
def factory(c):
 os.environ.pop("GLM_GROUP_FAULT_STATE_PATH",None);os.environ.update(c["environment"]);return create_app()
async def verify():
 app=factory(config);place=app.state.placement;idx=app.state.response_owner_index;bindings={}
 try:
  for key,size in[("D0",5),("D1",81932)]:
   lease=await place.acquire_for_owner(64,size,None);assert lease.replica.key==key
   ident="resp_independent_CPU_"+key;idx.bind(ident,lease);bindings[key]=idx.resolve(ident);await place.release(lease,backend_failure=False)
 finally:idx.close();await place.close()
 app=factory(config);place=app.state.placement;idx=app.state.response_owner_index
 try:
  for key,size in[("D0",81932),("D1",5)]:
   assert idx.resolve("resp_independent_CPU_"+key)==bindings[key]
   lease=await place.acquire_for_owner(64,size,bindings[key]);assert lease.replica.key==key;await place.release(lease,backend_failure=False)
 finally:idx.close();await place.close()
 contracts.append("same_epoch_restart_native_owner_beats_shape_preference")
 app=factory(new);place=app.state.placement;idx=app.state.response_owner_index
 try:
  try:await place.acquire_for_owner(64,5,idx.resolve("resp_independent_CPU_D1"))
  except RuntimeError as e:assert str(e)=="native response owner unavailable; no cross-owner replay"
  else:raise AssertionError("retired native D1 bound request accepted")
  assert not place.leases
  lease=await place.acquire_for_owner(64,81932,idx.resolve("resp_independent_CPU_D0"));assert lease.replica.key=="D0";await place.release(lease,backend_failure=False)
 finally:idx.close();await place.close()
 contracts.append("changed_D1_epoch_rejects_before_lease_D0_state_retained")
asyncio.run(verify())
for kind in["environment","evidence_hash"]:
 bad=json.loads(output.read_text())
 if kind=="environment":bad["environment"]["GLM_REPLICAS"]="[]"
 else:bad["resident_evidence"]["native_engines_resident.json"]["sha256"]="0"*64
 path=j/("tampered_"+kind+".json");path.write_text(json.dumps(bad))
 try:checked_config(path)
 except ValueError:contracts.append("tampered_"+kind+"_rejected")
 else:raise AssertionError(kind+"tampering accepted")
z=subprocess.run([sys.executable,str(runtime/"native_engines_service_entry.py"),"--help"],capture_output=True);z.check_returncode();(j/"entry_help.stdout").write_bytes(z.stdout);assert b"--config"in z.stdout;contracts.append("actual_native_engines_entry_CLI_help")
# One genuine recorded coupled131 engine is also accepted through the same compiler.
old=j.parents[1]/"runs/GLM-RUN-0131"
coupled=dict(id="coupled131",replica_id="D0",**{key:json.loads((old/name).read_text())for key,name in[("plans","standalone_launch.json"),("roots","standalone_root_identities.json"),("members","standalone_native_members.json")]})
single=compile_config(dict(engines=[coupled],placement=dict(kind="active_count")),j/"single_fixture_state");assert single["engine_geometry"]["coupled131"]["world_size"]==32and len(single["native_domains"])==1;contracts.append("recorded_coupled131_single_domain_supported")
print(json.dumps(dict(event="native_engines_CPU_valid",contracts=contracts,fixture=True,native_requests=0,NPU_workers_started=0,models=0,render_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),limits=["Syntheticlocal engine process/CLI evidence; no physicalengine startup/statefault/replication proof","One recorded coupled131 topology compiler check isread-only; actualtwoindependent localGPUconfigs stillneedCPU/nativeE2E","ActualV11factory/nativeaffinity/restart/render/entryCLI used, no listener/nativeHTTP/model operations"])))

import importlib.util
spec=importlib.util.spec_from_file_location("base_native_composer",j/"base_native_engines_service_config.py");base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
assert compile_config(material,state)==base.compile_config(material,state)
checks=["legacy_shape_config_unchanged"]
for kind in["active_count","shape_split"]:
 m=copy.deepcopy(material)
 if kind=="active_count":m["placement"]=dict(kind=kind)
 assert compile_config(m,state)==base.compile_config(m,state)
checks.append("legacy_active_config_unchanged")
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def hint_file(name,epoch,decode=10,prefill=100):
 f=j/(name+".json");f.write_text(json.dumps(dict(schema_version=1,native_owner_epoch=epoch,cohort_id="CPU-fixture",decode_tps=decode,prefill_bytes_per_s=prefill,method="CPU fixture; notobservedhardware",sources=[ref(j/"native_CPU_argv.json")],limitations=["CPU synthetic rate only; not native performance"])))
 return ref(f)
work=copy.deepcopy(material);work["placement"]=dict(kind="work_seconds")
for e in work["engines"]:
 epoch=next(v["epoch"]for v in config["native_domains"]if v["id"]==e["id"]);e["routing_hint"]=hint_file(e["id"]+"_hint",epoch)
wc=compile_config(work,j/"work_state");assert wc["native_domains"]==config["native_domains"]and len(wc["routing_hints"])==2
assert all(v["decode_tps"]==10 and v["prefill_bytes_per_s"]==100for v in json.loads(wc["environment"]["GLM_REPLICAS"]))
checks.append("explicit_work_seconds_rates_preserve_native_domains")
for kind in["bool_decode","nan_decode","negative_prefill","zero_prefill","stale_epoch","missing_method","missing_sources","missing_limitations","bad_hash","bad_size","source_hash","hints_on_active_count","unexpected_placement","unknown_engine_field"]:
 m=copy.deepcopy(work);e=m["engines"][0];h=json.loads(Path(e["routing_hint"]["path"]).read_text());f=j/("bad_"+kind+".json")
 if kind=="bool_decode":h["decode_tps"]=True
 elif kind=="nan_decode":h["decode_tps"]=float("nan")
 elif kind=="negative_prefill":h["prefill_bytes_per_s"]=-1
 elif kind=="zero_prefill":h["prefill_bytes_per_s"]=0
 elif kind=="stale_epoch":h["native_owner_epoch"]="0"*64
 elif kind=="missing_method":h["method"]=""
 elif kind=="missing_sources":h["sources"]=[]
 elif kind=="missing_limitations":h["limitations"]=[]
 elif kind=="source_hash":h["sources"][0]["sha256"]="0"*64
 f.write_text(json.dumps(h));e["routing_hint"]=ref(f)
 if kind=="bad_hash":e["routing_hint"]["sha256"]="0"*64
 elif kind=="bad_size":e["routing_hint"]["bytes"]+=1
 elif kind=="hints_on_active_count":m["placement"]=dict(kind="active_count")
 elif kind=="unexpected_placement":m["placement"]["rates"]="invented"
 elif kind=="unknown_engine_field":e["capacity"]="invented"
 try:compile_config(m,j/"bad_state")
 except ValueError:checks.append(kind+"_rejected")
 else:raise AssertionError(kind+"accepted")
changed=copy.deepcopy(work);changed["engines"][1]["members"]["node0"]["owned_targets"][0]["identity"]["start_ticks"]="999"
try:compile_config(changed,j/"stale_state")
except ValueError:checks.append("native_epoch_change_requires_fresh_hint")
else:raise AssertionError("stalehint accepted afternativeworkerreplacement")
async def new_verify():
 empty=copy.deepcopy(work)
 for e in empty["engines"]:e.pop("routing_hint")
 c=compile_config(empty,j/"fallback_state");app=factory(c);place=app.state.placement;idx=app.state.response_owner_index
 try:
  assert all(not x["work_ranking_calibrated"]for x in await place.snapshot())
  a=await place.acquire(4096,500);b=await place.acquire(64,500);assert {a.replica.key,b.replica.key}=={"D0","D1"}
  await place.release(a);await place.release(b)
 finally:idx.close();await place.close()
 checks.append("unhinted_work_policy_all_count_fallback")
 partial=copy.deepcopy(work);partial["engines"][1].pop("routing_hint")
 c=compile_config(partial,j/"partial_state");app=factory(c);place=app.state.placement;idx=app.state.response_owner_index
 try:
  assert all(not x["work_ranking_calibrated"]for x in await place.snapshot())
  a=await place.acquire(4096,500);b=await place.acquire(64,500);assert {a.replica.key,b.replica.key}=={"D0","D1"}
  await place.release(a);await place.release(b)
 finally:idx.close();await place.close()
 checks.append("one_missing_hint_all_count_fallback")
 app=factory(wc);place=app.state.placement;idx=app.state.response_owner_index
 try:
  assert all(x["work_ranking_calibrated"]for x in await place.snapshot())
  a=await place.acquire_for_owner(4096,500,None);key=a.replica.key;idx.bind("resp_work_CPU_owner",a);binding=idx.resolve("resp_work_CPU_owner");await place.release(a)
 finally:idx.close();await place.close()
 alt=copy.deepcopy(work)
 for e in alt["engines"]:
  epoch=next(v["epoch"]for v in config["native_domains"]if v["id"]==e["id"])
  e["routing_hint"]=hint_file(e["id"]+"_changed_hint",epoch,decode=1000 if e["replica_id"]!=key else 1,prefill=1000 if e["replica_id"]!=key else 1)
 c=compile_config(alt,j/"work_state");assert c["native_domains"]==wc["native_domains"];app=factory(c);place=app.state.placement;idx=app.state.response_owner_index
 try:
  assert idx.resolve("resp_work_CPU_owner")==binding
  a=await place.acquire_for_owner(1,1,binding);assert a.replica.key==key;await place.release(a)
  a=await place.acquire(1,1);assert a.replica.key!=key;await place.release(a)
 finally:idx.close();await place.close()
 checks.append("native_owner_affinity_beats_changed_work_hint_preference")
asyncio.run(new_verify())
# Reuse verified138 isolated short32 and NEWcold81932 diagnostics; derive actual coarse hints.
point=j.parents[1];r=point/"runs/GLM-RUN-0138";audit=point/"jobs/AUDIT-RUN138-20261003T0302Z/reduction.json"
assert ref(audit)["sha256"]=="962c487e12e63bfd6d9760d82762b21f6c44e7f427abbed96276a911e288b256"
pilot=json.loads((r/"pilot_summary.json").read_text());actual=json.loads((point/"runs/GLM-RUN-0137/native_engines_resident.json").read_text());realc=compile_config(actual,j/"real_epoch_state");hintrefs={}
for owner in["D0","D1"]:
 short=next(x for x in pilot["requests"]if x["name"]=="short_"+owner);cold=next(x for x in pilot["requests"]if x["name"]=="full_"+owner)
 sources=[ref(audit),ref(r/"pilot_summary.json")]
 for row in[short,cold]:
  assert row["completed"]and row["contract"]["done"]and row["contract"]["finish_reasons"]=={"0":"length"}
  body=r/(row["name"]+".body.json");assert ref(body)["sha256"]==row["request_body_sha256"];assert ref(Path(row["wire"]["path"]))==row["wire"];sources.extend([ref(body),row["wire"]])
 assert short["usage"]["completion_tokens"]==32and cold["usage"]["prompt_tokens"]==81932and cold["usage"]["completion_tokens"]==64
 epoch=next(x["epoch"]for x in realc["native_domains"]if owner in x["members"])
 f=j/("observed_"+owner+".json");f.write_text(json.dumps(dict(schema_version=1,native_owner_epoch=epoch,cohort_id="GLM-COHORT-0137",decode_tps=31/(short["wall_s"]-short["ttft_s"]),prefill_bytes_per_s=(r/(cold["name"]+".body.json")).stat().st_size/cold["ttft_s"],method="Run138 isolated short32 residual31/(wall-first-content), cold81932 serializedHTTPbodybytes/first-content-time",sources=sources,limitations=["Coarse observed hints include HTTP/SDK/nativeMTP3 transport, not GPUtime or upperbound","No prediction of batch speedup, progress, shared interference or native background work","Calibration applies recorded137native epochs/cohort; no general performance or qualityclaim"]),indent=2)+"\n");hintrefs[owner]=ref(f)
actual["placement"]=dict(kind="work_seconds")
for e in actual["engines"]:e["routing_hint"]=hintrefs[e["replica_id"]]
ac=compile_config(actual,j/"real_epoch_state");assert ac["native_domains"]==realc["native_domains"];checks.append("actual138_observation_hints_bind_137_epochs_without_inference")
resident=j/"work_resident_fixture";resident.mkdir();(resident/"native_engines_resident.json").write_text(json.dumps(actual));z=subprocess.run([sys.executable,str(j/"native_engines_service_config.py"),"--resident-dir",str(resident),"--state-dir",str(j/"real_epoch_state"),"--output",str(j/"work_rendered_config.json")],capture_output=True);z.check_returncode();assert checked_config(j/"work_rendered_config.json")[0]==dict(ac,resident_evidence={"native_engines_resident.json":dict(path=str(resident/"native_engines_resident.json"),sha256=ref(resident/"native_engines_resident.json")["sha256"])})
checks.append("actual_candidate_work_render_and_checked_source_artifacts")
print(json.dumps(dict(event="work_hint_CPU_valid",contracts=checks,legacy_contracts=len(contracts),observed_hint_refs=hintrefs,native_requests=0,NPU_workers_started=0,models=0,source=ref(j/"native_engines_service_config.py"))))
