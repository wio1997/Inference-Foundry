import asyncio,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
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
z=subprocess.run([sys.executable,str(runtime/"native_engines_service_config.py"),"--resident-dir",str(fixture),"--state-dir",str(state),"--output",str(output)],capture_output=True);z.check_returncode();(j/"render.stdout").write_bytes(z.stdout);(j/"render.stderr").write_bytes(z.stderr);assert checked_config(output)[0]["native_domains"]==config["native_domains"];contracts.append("actual_render_and_checked_provenance")
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
